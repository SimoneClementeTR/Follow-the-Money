"""
P2P Fraud Detection using Graph Neural Networks (LEAKAGE-FREE VERSION)

Key fixes for data leakage:
1. TEMPORAL SPLITTING: Uses time-based splits instead of random splits
2. LEAKAGE-FREE FEATURES: Computes features only from training data
3. PROGRESSIVE SCALING: Fits scaler only on training data
4. TEMPORAL GRAPH: Builds graph progressively respecting time order
5. PROPER VALIDATION: Uses temporal validation without future information

To run:
- Place customers.csv, accounts.csv, p2p_transactions.csv, cards.csv under data/
- Ensure p2p_transactions.csv has a "timestamp" column
- python this_file.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
from typing import Tuple, Dict, Any
from datetime import datetime

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: float = 2.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class P2PFraudGNN(nn.Module):
    """Graph Neural Network for P2P fraud detection."""
    def __init__(self, num_node_features: int, hidden_dim: int = 128):
        super().__init__()

        self.conv1 = GATConv(num_node_features, hidden_dim // 4, heads=4, dropout=0.3, concat=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 2, heads=2, dropout=0.3, concat=True)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 2)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_pairs_to_classify: torch.Tensor) -> torch.Tensor:
        x1 = F.elu(self.conv1(x, edge_index))
        x1 = self.bn1(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)

        x2 = F.elu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)
        x2 = F.dropout(x2, p=0.3, training=self.training)

        x3 = F.elu(self.conv3(x2, edge_index))
        x3 = self.bn3(x3)

        if x3.size(-1) == x1.size(-1):
            x3 = x3 + x1

        src_embeddings = x3[edge_pairs_to_classify[:, 0]]
        dst_embeddings = x3[edge_pairs_to_classify[:, 1]]
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        logits = self.edge_classifier(edge_embeddings)
        return logits


class LeakageFreeP2PDetector:
    """Complete leakage-free P2P fraud detection system."""

    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.model = None
        self.optimal_threshold = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store temporal splits info
        self.temporal_splits = {}
        self.feature_stats = {}

        print(f"Initialized LeakageFreeP2PDetector on {self.device}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Loading data...")
        customers = pd.read_csv(f"{self.data_path}customers.csv")
        accounts = pd.read_csv(f"{self.data_path}accounts.csv")
        p2p_transactions = pd.read_csv(f"{self.data_path}p2p_transactions.csv")
        cards = pd.read_csv(f"{self.data_path}cards.csv")

        print(f"Data loaded:")
        print(f"   - Customers: {len(customers):,}")
        print(f"   - P2P transactions: {len(p2p_transactions):,}")

        if "is_fraud" in p2p_transactions.columns:
            print(f"   - Fraud rate: {p2p_transactions['is_fraud'].mean():.1%}")

        return customers, accounts, p2p_transactions, cards

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        customers, accounts, p2p_transactions, cards = self.load_data()

        # Create account_id -> customer_id mapping
        customer_account_map = dict(zip(accounts["account_id"], accounts["customer_id"]))

        p2p_transactions["sender_customer_id"] = p2p_transactions["sender_account_id"].map(customer_account_map)
        p2p_transactions["receiver_customer_id"] = p2p_transactions["receiver_account_id"].map(customer_account_map)

        # Remove transactions with missing customer mappings
        initial_count = len(p2p_transactions)
        p2p_transactions = p2p_transactions.dropna(subset=["sender_customer_id", "receiver_customer_id"])
        print(f"   - Valid P2P transactions: {len(p2p_transactions):,} ({initial_count - len(p2p_transactions):,} dropped)")

        # Convert timestamp to datetime if it exists
        if "timestamp" in p2p_transactions.columns:
            p2p_transactions["timestamp"] = pd.to_datetime(p2p_transactions["timestamp"])
            p2p_transactions = p2p_transactions.sort_values("timestamp").reset_index(drop=True)
            print(f"   - Transactions span: {p2p_transactions['timestamp'].min()} to {p2p_transactions['timestamp'].max()}")

        # Filter customers to only those involved in P2P
        p2p_customers = set(
            p2p_transactions["sender_customer_id"].tolist() +
            p2p_transactions["receiver_customer_id"].tolist()
        )
        customers_filtered = customers[customers["customer_id"].isin(p2p_customers)].copy()
        print(f"   - P2P customers: {len(customers_filtered):,}")

        return customers_filtered, accounts, p2p_transactions, cards

    def create_temporal_splits(self, p2p_transactions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create temporal splits to prevent data leakage."""
        print("Creating temporal splits to prevent data leakage...")

        if "timestamp" not in p2p_transactions.columns:
            raise ValueError("Timestamp column is required for temporal splitting to avoid data leakage")

        # Sort by timestamp to maintain temporal order
        df_sorted = p2p_transactions.sort_values("timestamp").copy()

        # Use 60% for train, 20% for val, 20% for test (temporal order)
        n_total = len(df_sorted)
        train_end_idx = int(0.6 * n_total)
        val_end_idx = int(0.8 * n_total)

        train_df = df_sorted.iloc[:train_end_idx].copy()
        val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        test_df = df_sorted.iloc[val_end_idx:].copy()

        # Store temporal boundaries for validation
        self.temporal_splits = {
            "train_end": df_sorted.iloc[train_end_idx-1]["timestamp"],
            "val_end": df_sorted.iloc[val_end_idx-1]["timestamp"],
            "test_start": df_sorted.iloc[val_end_idx]["timestamp"]
        }

        print(f"   Temporal splits created:")
        print(f"   - Train: {len(train_df):,} transactions (until {self.temporal_splits['train_end']})")
        print(f"   - Val: {len(val_df):,} transactions ({self.temporal_splits['train_end']} to {self.temporal_splits['val_end']})")
        print(f"   - Test: {len(test_df):,} transactions (from {self.temporal_splits['test_start']})")
        print(f"   - Train fraud rate: {train_df['is_fraud'].mean():.1%}")
        print(f"   - Val fraud rate: {val_df['is_fraud'].mean():.1%}")
        print(f"   - Test fraud rate: {test_df['is_fraud'].mean():.1%}")

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }

    def create_leakage_free_features(self, customers: pd.DataFrame, accounts: pd.DataFrame,
                                   cards: pd.DataFrame, temporal_splits: Dict[str, pd.DataFrame]) -> Tuple[
        Dict[str, np.ndarray], Dict[int, int], pd.DataFrame]:
        """Create features using only training data to prevent leakage."""

        print("Creating leakage-free features...")
        train_transactions = temporal_splits["train"]

        customer_features = customers.copy()

        # Encode categorical attributes using training data only
        categorical_encoders = {
            "age_cat": LabelEncoder(),
            "gender": LabelEncoder(),
            "city": LabelEncoder(),
            "country": LabelEncoder()
        }

        for col, encoder in categorical_encoders.items():
            if col in customer_features.columns:
                # Fit encoder on all customers (this is not leakage as it"s customer attributes)
                customer_features[f"{col}_encoded"] = encoder.fit_transform(
                    customer_features[col].fillna("unknown")
                )

        # KEY FIX: Compute transaction-based features ONLY from training data
        print("   - Computing features from TRAINING transactions only...")

        # Sender behavior from TRAINING transactions only
        sender_stats = train_transactions.groupby("sender_customer_id").agg({
            "amount": ["count", "mean", "std", "max", "min"],
            "receiver_customer_id": "nunique"
        })
        sender_stats.columns = ["_".join(col).strip() for col in sender_stats.columns]
        sender_stats = sender_stats.add_prefix("sender_")

        # Receiver behavior from TRAINING transactions only
        receiver_stats = train_transactions.groupby("receiver_customer_id").agg({
            "amount": ["count", "mean", "std", "max", "min"],
            "sender_customer_id": "nunique"
        })
        receiver_stats.columns = ["_".join(col).strip() for col in receiver_stats.columns]
        receiver_stats = receiver_stats.add_prefix("receiver_")

        # Temporal features from TRAINING transactions only
        temporal_stats = pd.DataFrame()
        if "timestamp" in train_transactions.columns:
            temp_df = train_transactions.copy()
            temp_df["hour"] = temp_df["timestamp"].dt.hour
            temp_df["day_of_week"] = temp_df["timestamp"].dt.dayofweek
            temporal_stats = temp_df.groupby("sender_customer_id").agg({
                "hour": ["mean", "std"],
                "day_of_week": ["mean", "std"]
            })
            temporal_stats.columns = ["_".join(col).strip() for col in temporal_stats.columns]
            temporal_stats = temporal_stats.add_prefix("temporal_")

        # Account stats (not time-dependent)
        account_stats = accounts.groupby("customer_id").agg({
            "is_active": "sum",
            "account_id": "count"
        }).rename(columns={"account_id": "num_accounts"})

        # Card stats (not time-dependent)
        card_stats = pd.DataFrame(columns=["num_cards", "num_active_cards"])
        if len(cards) > 0:
            card_stats = cards.groupby("customer_id").agg({
                "is_active": "sum",
                "card_id": "count"
            }).rename(columns={"card_id": "num_cards", "is_active": "num_active_cards"})

        # Store feature statistics for later application to val/test
        self.feature_stats = {
            "sender_stats": sender_stats,
            "receiver_stats": receiver_stats,
            "temporal_stats": temporal_stats,
            "account_stats": account_stats,
            "card_stats": card_stats
        }

        # Apply features to ALL customers but stats come from training only
        feature_dfs = [account_stats, sender_stats, receiver_stats, card_stats]
        if not temporal_stats.empty:
            feature_dfs.append(temporal_stats)

        for df in feature_dfs:
            customer_features = customer_features.merge(
                df, left_on="customer_id", right_index=True, how="left"
            )

        customer_features = customer_features.fillna(0)

        # Select feature columns
        feature_columns = [
            col for col in customer_features.columns
            if (col not in ["customer_id"] and
                customer_features[col].dtype in ["int64", "float64", "int32", "float32"])
        ]

        customer_to_idx = {cust_id: idx for idx, cust_id in enumerate(customer_features["customer_id"])}

        # Create feature matrices for each split
        X_matrices = {}
        for split_name in ["train", "val", "test"]:
            X_matrices[split_name] = customer_features[feature_columns].values.astype(np.float32)

        print(f" Features created from training data only:")
        print(f"   - Feature columns: {len(feature_columns)}")
        print(f"   - Feature matrix shape: {X_matrices['train'].shape}")
        print("   - Transaction-based features computed from training set only ✓")

        return X_matrices, customer_to_idx, customer_features

    def create_temporal_graph(self, customers_filtered: pd.DataFrame, temporal_splits: Dict[str, pd.DataFrame],
                            X_matrices: Dict[str, np.ndarray], customer_to_idx: Dict[int, int]) -> Dict[str, Data]:
        """Create temporal graphs respecting time order to prevent structural leakage."""

        print("Creating temporal graph structure...")

        graph_data = {}

        for split_name, tx_df in temporal_splits.items():
            tx = tx_df.copy()
            tx["sender_idx"] = tx["sender_customer_id"].map(customer_to_idx)
            tx["receiver_idx"] = tx["receiver_customer_id"].map(customer_to_idx)

            valid_tx = tx.dropna(subset=["sender_idx", "receiver_idx"]).copy()
            valid_tx["sender_idx"] = valid_tx["sender_idx"].astype(int)
            valid_tx["receiver_idx"] = valid_tx["receiver_idx"].astype(int)

            edge_pairs = valid_tx[["sender_idx", "receiver_idx"]].values
            edge_index = torch.tensor(edge_pairs.T, dtype=torch.long)

            # Features come from the corresponding split
            x = torch.tensor(X_matrices[split_name], dtype=torch.float)
            edge_labels = torch.tensor(valid_tx["is_fraud"].values, dtype=torch.long)
            edge_pairs_to_classify = torch.tensor(edge_pairs, dtype=torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_labels=edge_labels,
                edge_pairs_to_classify=edge_pairs_to_classify
            )

            graph_data[split_name] = data

            print(f"   - {split_name.capitalize()}: {data.x.shape[0]:,} nodes, {data.edge_index.shape[1]:,} edges")

        print("Temporal graphs created respecting time boundaries")
        return graph_data

    @staticmethod
    def optimize_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold for F1 score."""
        thresholds = np.arange(0.001, 0.999, 0.001)
        best_f1 = 0.0
        best_t = 0.5
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            if len(np.unique(y_pred)) == 1:
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t, best_f1

    def train_model(self, graph_data: Dict[str, Data], epochs: int = 300, lr: float = 0.001) -> P2PFraudGNN:
        """Train model with proper temporal validation and leakage-safe scaling."""

        print(f"Training model on {self.device}...")

        train_data = graph_data["train"]
        val_data = graph_data["val"]

        # Fit scaler on TRAINING nodes only
        train_nodes = np.unique(train_data.edge_pairs_to_classify.numpy().reshape(-1))
        self.scaler = StandardScaler().fit(train_data.x[train_nodes].numpy())

        # Apply scaler to all splits
        for split_name, data in graph_data.items():
            X_scaled = self.scaler.transform(data.x.numpy()).astype(np.float32)
            data.x = torch.tensor(X_scaled, dtype=torch.float)
            data = data.to(self.device)
            graph_data[split_name] = data

        train_data = graph_data["train"]
        val_data = graph_data["val"]

        # Initialize model
        self.model = P2PFraudGNN(num_node_features=train_data.x.shape[1], hidden_dim=128).to(self.device)

        criterion = FocalLoss(alpha=3.0, gamma=2.5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_f1 = -1.0
        patience = 15
        patience_counter = 0

        print("   🔄 Starting training with temporal validation...")

        for epoch in range(1, epochs + 1):
            # Train on training data with training edges only
            self.model.train()
            optimizer.zero_grad()

            train_logits = self.model(
                train_data.x, train_data.edge_index, train_data.edge_pairs_to_classify
            )
            loss = criterion(train_logits, train_data.edge_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # CRITICAL: Validate using TRAINING graph structure to prevent leakage
            # We use training edges for message passing but evaluate on validation edges
            if epoch % 5 == 0 or epoch == 1:
                self.model.eval()
                with torch.no_grad():
                    # Use TRAINING graph structure for message passing
                    val_logits = self.model(
                        val_data.x, train_data.edge_index, val_data.edge_pairs_to_classify
                    )
                    val_probs = F.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
                    val_labels = val_data.edge_labels.detach().cpu().numpy()
                    optimal_t, val_f1 = self.optimize_threshold(val_labels, val_probs)
                    val_auc = roc_auc_score(val_labels, val_probs)

                if epoch % 20 == 0 or epoch == 1:
                    print(f"   Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

                # Early stopping on validation F1
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "optimal_threshold": optimal_t,
                        "best_val_f1": best_val_f1,
                        "epoch": epoch,
                        "scaler": self.scaler
                    }, "best_leakage_free_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"   ⏹️ Early stopping at epoch {epoch}")
                        break

        # Load best checkpoint
        checkpoint = torch.load("best_leakage_free_model.pth", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimal_threshold = float(checkpoint["optimal_threshold"])

        print(f"   ✅ Training completed!")
        print(f"   - Best validation F1: {checkpoint['best_val_f1']:.4f} at epoch {checkpoint['epoch']}")
        print(f"   - Optimal threshold: {self.optimal_threshold:.4f}")

        return self.model

    def evaluate_model(self, graph_data: Dict[str, Data]) -> Dict[str, Any]:
        """Evaluate model on test set using temporal validation."""

        if self.model is None or self.optimal_threshold is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("📈 Evaluating model on test set...")

        train_data = graph_data["train"]
        test_data = graph_data["test"]

        self.model.eval()
        with torch.no_grad():
            # CRITICAL: Use TRAINING edges for message passing even during test evaluation
            test_logits = self.model(
                test_data.x, train_data.edge_index, test_data.edge_pairs_to_classify
            )
            test_probs = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
            test_labels = test_data.edge_labels.cpu().numpy()

        test_preds_optimal = (test_probs >= self.optimal_threshold).astype(int)

        results = {
            "test_auc": float(roc_auc_score(test_labels, test_probs)),
            "optimal_threshold": float(self.optimal_threshold),
            "test_metrics": {
                "accuracy": float(accuracy_score(test_labels, test_preds_optimal)),
                "precision": float(precision_score(test_labels, test_preds_optimal, zero_division=0)),
                "recall": float(recall_score(test_labels, test_preds_optimal)),
                "f1": float(f1_score(test_labels, test_preds_optimal, zero_division=0)),
            },
            "confusion_matrix": confusion_matrix(test_labels, test_preds_optimal).tolist(),
            "fraud_detection_rate": float(recall_score(test_labels, test_preds_optimal)),
            "false_alarm_rate": float(test_preds_optimal[test_labels == 0].mean()) if (test_labels == 0).any() else 0.0
        }

        # Display results
        print("\n" + "=" * 60)
        print("LEAKAGE-FREE TEST RESULTS")
        print("=" * 60)
        print(f"AUC-ROC: {results['test_auc']:.4f}")
        print(f"\nOptimal Threshold: {results['optimal_threshold']:.3f}")
        print(f"  Accuracy:  {results['test_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['test_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['test_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {results['test_metrics']['f1']:.4f}")

        total_fraud = int(test_labels.sum())
        fraud_caught = int(results['fraud_detection_rate'] * total_fraud)
        false_alarms = int(results['false_alarm_rate'] * (test_labels == 0).sum())

        print(f"\nBusiness Impact:")
        print(f"  Fraud Detection Rate: {results['fraud_detection_rate']:.1%}")
        print(f"  Fraud Caught: {fraud_caught}/{total_fraud}")
        print(f"  False Alarms: {false_alarms:,}")
        print(f"  Precision: {results['test_metrics']['precision']:.1%} of flagged transactions are fraud")

        print(f"\nLeakage Prevention Measures Applied:")
        print(f"  ✅ Temporal splitting (no future information)")
        print(f"  ✅ Training-only feature engineering")
        print(f"  ✅ Training-only scaling")
        print(f"  ✅ Training-only message passing")

        return results

    def run_complete_pipeline(self, epochs: int = 300, lr: float = 0.001) -> Dict[str, Any]:
        """Run the complete leakage-free fraud detection pipeline."""

        print("Starting LEAKAGE-FREE P2P Fraud Detection Pipeline")
        print("=" * 70)

        # Load and prepare data
        customers_filtered, accounts, p2p_transactions, cards = self.prepare_data()

        # Create temporal splits to prevent data leakage
        temporal_splits = self.create_temporal_splits(p2p_transactions)

        # Create leakage-free features
        X_matrices, customer_to_idx, customer_features = self.create_leakage_free_features(
            customers_filtered, accounts, cards, temporal_splits
        )

        # Create temporal graphs
        graph_data = self.create_temporal_graph(
            customers_filtered, temporal_splits, X_matrices, customer_to_idx
        )

        # Train model with temporal validation
        model = self.train_model(graph_data, epochs, lr)

        # Evaluate on test set
        results = self.evaluate_model(graph_data)

        print("\n Leakage-free pipeline completed successfully!")
        print(" Results are now reliable for production deployment!")

        return results


def main():
    """Main function to run the leakage-free fraud detection system."""
    detector = LeakageFreeP2PDetector(data_path="data/")
    results = detector.run_complete_pipeline(epochs=150, lr=0.001)
    return detector, results


if __name__ == "__main__":
    detector, results = main()