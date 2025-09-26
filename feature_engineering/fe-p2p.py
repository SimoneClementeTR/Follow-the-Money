#!/usr/bin/env python3
"""
Feature engineering script for P2P fraud dataset.

Inputs (CSV) expected in --data-dir:
- p2p_transactions.csv  (must contain: timestamp, amount, currency,
                         sender_account_id, receiver_account_id, is_fraud)
- accounts.csv          (must contain: account_id, customer_id, open_date, is_active)
- customers.csv         (should contain: customer_id, age_cat, gender, city)

Output:
- engineered_features.parquet  (features + optional label "is_fraud"; helper keys dropped)

Usage:
  uv run feature_engineering/fe-p2p.py --data-dir ./data --out .data/engineered_features.parquet
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------- utils -------------------------
def to_utc(series: pd.Series) -> pd.Series:
    """Coerce any datetime-like series to tz-aware UTC (handles naive/aware)."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def safe_bool(series: pd.Series) -> pd.Series:
    """Cast mixed/str boolean-like column to pandas BooleanDtype then to int."""
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype.name == "boolean":
        return series.fillna(False).astype(int)
    s = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    )
    return s.fillna(False).astype(int)


# ---------------------- feature engineering ----------------------
def build_static_dims(accounts: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Join customer attributes onto accounts to get static account dims."""
    # Standardize column names expected downstream
    customers = customers.rename(
        columns={
            "customer_id": "customer_id",
            "age_cat": "cust_age_cat",
            "gender": "cust_gender",
            "city": "cust_city",
        }
    )
    acc = accounts.merge(
        customers.add_prefix("cust_").rename(columns={"cust_customer_id": "cust_customer_id"}),
        left_on="customer_id",
        right_on="cust_customer_id",
        how="left",
    ).drop(columns=["cust_customer_id"], errors="ignore")

    # Clean categoricals
    for col in ["cust_age_cat", "cust_gender", "cust_city"]:
        if col in acc.columns:
            acc[col] = acc[col].astype(str).fillna("unknown").replace({"nan": "unknown"})
        else:
            acc[col] = "unknown"

    # Normalize boolean is_active; keep original boolean form here
    if "is_active" in acc.columns:
        acc["is_active"] = acc["is_active"]
    else:
        acc["is_active"] = False

    return acc


def build_features(p2p: pd.DataFrame, acc_dims: pd.DataFrame) -> pd.DataFrame:
    """Create all requested features; returns a modeling table with one-hot encoded categoricals."""
    # Sort chronologically (tz-aware)
    df = p2p.sort_values("timestamp").copy()

    # Attach sender and receiver dimensions
    s = acc_dims.add_prefix("s_")
    r = acc_dims.add_prefix("r_")
    df = df.merge(s, left_on="sender_account_id", right_on="s_account_id", how="left")
    df = df.merge(r, left_on="receiver_account_id", right_on="r_account_id", how="left")

    # Transaction-level features
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    df["same_currency"] = (df["currency"].astype(str) == "CHF").astype(int)

    # Relationship keys and counts (helper keys will be dropped later)
    df["pair_key"] = df["sender_account_id"].astype(str) + "|" + df["receiver_account_id"].astype(str)
    df["rev_pair_key"] = df["receiver_account_id"].astype(str) + "|" + df["sender_account_id"].astype(str)
    df["prior_pair_txn_count"] = df.groupby("pair_key").cumcount()
    df["prior_rev_pair_txn_count"] = df.groupby("rev_pair_key").cumcount()

    # Sender behavior/history
    df["s_prior_send_count"] = df.groupby("sender_account_id").cumcount()
    df["s_cum_amount"] = (
        df.groupby("sender_account_id")["amount"].cumsum().shift(1).fillna(0)
    )
    denom = df["s_prior_send_count"].replace(0, np.nan)
    df["s_prior_avg_amount"] = (df["s_cum_amount"] / denom).fillna(0)
    df["s_amt_dev_from_avg"] = df["amount"] - df["s_prior_avg_amount"]

    # Receiver behavior/history
    df["r_prior_recv_count"] = df.groupby("receiver_account_id").cumcount()

    # Same-customer and same-city
    df["same_customer"] = (df["s_customer_id"] == df["r_customer_id"]).astype(int)
    df["same_city"] = (df["s_cust_city"].astype(str) == df["r_cust_city"].astype(str)).astype(int)

    # Account ages and active flags (ensure tz-safe subtraction)
    for side in ["s_", "r_"]:
        open_col = f"{side}open_date"
        if open_col in df.columns:
            age_days = (df["timestamp"].dt.normalize() - df[open_col]).dt.days
            df[f"{side}account_age_days"] = age_days.clip(lower=0).fillna(0)
        else:
            df[f"{side}account_age_days"] = 0

        active_col = f"{side}is_active"
        df[active_col] = safe_bool(df[active_col]) if active_col in df.columns else 0

    # Keep target if present (not used as a feature but kept in output file)
    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].astype(int)

    # One-hot encode requested categoricals
    cat_cols = [
        "currency",
        "s_cust_age_cat", "s_cust_gender", "s_cust_city",
        "r_cust_age_cat", "r_cust_gender", "r_cust_city",
    ]
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].astype(str).fillna("unknown").replace({"nan": "unknown"})

    df_ohe = pd.get_dummies(
        df,
        columns=cat_cols,
        prefix=cat_cols,
        prefix_sep="=",
        dtype=np.uint8,
    )

    # Drop helper and identifier columns not meant for modeling
    # drop_cols = [
    #     "p2p_id",
    #     "pair_key", "rev_pair_key",
    #     "s_account_id", "r_account_id",
    #     "sender_account_id", "receiver_account_id",
    #     # raw dates often excluded from modeling; we keep timestamp for reference in output
    #     # but not required as a feature input
    # ]
    # df_ohe = df_ohe.drop(columns=[c for c in drop_cols if c in df_ohe.columns], errors="ignore")

    return df_ohe


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Build engineered features and save to Parquet.")
    ap.add_argument("--data-dir", type=str, default="./data", help="Directory containing input CSVs")
    ap.add_argument("--out", type=str, default="engineered_features.parquet", help="Output Parquet path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    # Load raw data
    p2p = pd.read_csv(data_dir / "p2p_transactions.csv")
    accounts = pd.read_csv(data_dir / "accounts.csv")
    customers = pd.read_csv(data_dir / "customers.csv")

    # Coerce datetimes to tz-aware UTC to avoid tz-naive/aware mix
    p2p["timestamp"] = to_utc(p2p["timestamp"])
    accounts["open_date"] = to_utc(accounts["open_date"])

    # Ensure expected columns exist with sensible types
    required_p2p = ["timestamp", "amount", "currency", "sender_account_id", "receiver_account_id"]
    for col in required_p2p:
        if col not in p2p.columns:
            raise ValueError(f"Missing column in p2p_transactions.csv: {col}")

    accounts.rename(columns={"account_id": "account_id"}, inplace=True)
    if "account_id" not in accounts.columns:
        # try common alternatives
        for alt in ["id", "acct_id", "accountId"]:
            if alt in accounts.columns:
                accounts.rename(columns={alt: "account_id"}, inplace=True)
                break
    if "account_id" not in accounts.columns:
        raise ValueError("accounts.csv must contain 'account_id' column.")

    # Build static dims and features
    acc_dims = build_static_dims(accounts, customers)
    features = build_features(p2p, acc_dims)

    # Save parquet
    out_path = Path(args.out)
    features.to_parquet(out_path, index=False)
    print(f"Wrote {out_path.resolve()} with shape {features.shape}")


if __name__ == "__main__":
    main()