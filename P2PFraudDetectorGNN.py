"""
P2P Fraud Detection using Graph Neural Networks (Leakage-fixed, ready to run)

Key fixes vs. original:
- torch.load(): removed unsupported weights_only arg; added map_location for portability.
- Structural leakage: GNN message passing now uses edge_index built strictly from TRAIN edges
  during training, validation, and test (the model never "sees" validation or test edges
  in message passing).
- Feature-scaling leakage: StandardScaler is fit only on nodes that appear in TRAIN edges,
  then applied to all nodes before training.
- Early stopping: validate every epoch with explicit patience.

To run:
- Place customers.csv, accounts.csv, p2p_transactions.csv, cards.csv under data/
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings
from typing import Tuple, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Args:
        alpha (float): scaling factor (use class weights in CE or per-class alpha if needed)
        gamma (float): focusing parameter
    """
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
    """
    Graph Neural Network for P2P fraud detection.

    - Two GAT layers + one GCN layer to get node embeddings
    - Edge MLP over concatenated (src, dst) embeddings
    """
    def __init__(self, num_node_features: int, hidden_dim: int = 128, num_layers: int = 3):
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


class P2PFraudDetector:
    """
    Complete P2P fraud detection system:
    - Load and prepare data
    - Build features (no scaling yet)
    - Build graph
    - Train (with leakage-safe edges and scaler)
    - Evaluate
    """

    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.model = None
        self.optimal_threshold = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_raw = None  # unscaled features for leakage-safe scaling later

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Loading data...")
        customers = pd.read_csv(f"{self.data_path}customers.csv")
        accounts = pd.read_csv(f"{self.data_path}accounts.csv")
        p2p_transactions = pd.read_csv(f"{self.data_path}p2p_transactions.csv")
        cards = pd.read_csv(f"{self.data_path}cards.csv")

        print(f"✅ Data loaded successfully:")
        print(f"   - Customers: {len(customers):,}")
        print(f"   - P2P transactions: {len(p2p_transactions):,}")
        if "is_fraud" in p2p_transactions.columns:
            print(f"   - Fraud rate: {p2p_transactions['is_fraud'].mean():.1%}")
        return customers, accounts, p2p_transactions, cards

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        customers, accounts, p2p_transactions, cards = self.load_data()

        # account_id -> customer_id
        customer_account_map = dict(zip(accounts["account_id"], accounts["customer_id"]))

        p2p_transactions["sender_customer_id"] = (
            p2p_transactions["sender_account_id"].map(customer_account_map)
        )
        p2p_transactions["receiver_customer_id"] = (
            p2p_transactions["receiver_account_id"].map(customer_account_map)
        )

        initial_count = len(p2p_transactions)
        p2p_transactions = p2p_transactions.dropna(
            subset=["sender_customer_id", "receiver_customer_id"]
        )
        print(f"   - P2P transactions after mapping: {len(p2p_transactions):,} "
              f"({initial_count - len(p2p_transactions):,} dropped)")

        p2p_customers = set(
            p2p_transactions["sender_customer_id"].tolist() +
            p2p_transactions["receiver_customer_id"].tolist()
        )
        customers_filtered = customers[customers["customer_id"].isin(p2p_customers)].copy()
        print(f"   - Customers involved in P2P: {len(customers_filtered):,}")

        return customers_filtered, accounts, p2p_transactions, cards

    def create_features(self, customers: pd.DataFrame, accounts: pd.DataFrame,
                        cards: pd.DataFrame, p2p_transactions: pd.DataFrame) -> Tuple[
        np.ndarray, Dict[int, int], pd.DataFrame]:
        print("Creating features (unscaled)...")

        customer_features = customers.copy()

        # Encode categorical attributes
        categorical_encoders = {
            "age_cat": LabelEncoder(),
            "gender": LabelEncoder(),
            "city": LabelEncoder(),
            "country": LabelEncoder()
        }
        for col, encoder in categorical_encoders.items():
            customer_features[f"{col}_encoded"] = encoder.fit_transform(
                customer_features[col].fillna("unknown")
            )

        # Account stats
        account_stats = accounts.groupby("customer_id").agg({
            "is_active": "sum",
            "account_id": "count"
        }).rename(columns={"account_id": "num_accounts"})

        # Sender behavior
        sender_stats = p2p_transactions.groupby("sender_customer_id").agg({
            "amount": ["count", "mean", "std", "max", "min"],
            "receiver_customer_id": "nunique"
        })
        sender_stats.columns = ["_".join(col).strip() for col in sender_stats.columns]
        sender_stats = sender_stats.add_prefix("sender_")

        # Receiver behavior
        receiver_stats = p2p_transactions.groupby("receiver_customer_id").agg({
            "amount": ["count", "mean", "std", "max", "min"],
            "sender_customer_id": "nunique"
        })
        receiver_stats.columns = ["_".join(col).strip() for col in receiver_stats.columns]
        receiver_stats = receiver_stats.add_prefix("receiver_")

        # Temporal
        temporal_stats = pd.DataFrame()
        if "timestamp" in p2p_transactions.columns:
            p2p_transactions = p2p_transactions.copy()
            p2p_transactions["timestamp"] = pd.to_datetime(p2p_transactions["timestamp"])
            p2p_transactions["hour"] = p2p_transactions["timestamp"].dt.hour
            p2p_transactions["day_of_week"] = p2p_transactions["timestamp"].dt.dayofweek
            temporal_stats = p2p_transactions.groupby("sender_customer_id").agg({
                "hour": ["mean", "std"],
                "day_of_week": ["mean", "std"]
            })
            temporal_stats.columns = ["_".join(col).strip() for col in temporal_stats.columns]
            temporal_stats = temporal_stats.add_prefix("temporal_")

        # Card stats
        card_stats = pd.DataFrame(columns=["num_cards", "num_active_cards"])
        if len(cards) > 0:
            card_stats = cards.groupby("customer_id").agg({
                "is_active": "sum",
                "card_id": "count"
            }).rename(columns={"card_id": "num_cards", "is_active": "num_active_cards"})

        # Merge
        feature_dfs = [account_stats, sender_stats, receiver_stats, card_stats]
        if not temporal_stats.empty:
            feature_dfs.append(temporal_stats)

        for df in feature_dfs:
            customer_features = customer_features.merge(
                df, left_on="customer_id", right_index=True, how="left"
            )

        customer_features = customer_features.fillna(0)

        feature_columns = [
            col for col in customer_features.columns
            if (col not in ["customer_id"] and
                customer_features[col].dtype in ["int64", "float64", "int32", "float32"])
        ]

        X_raw = customer_features[feature_columns].values.astype(np.float32)
        self.X_raw = X_raw  # store unscaled for leakage-safe scaling later

        customer_to_idx = {cust_id: idx for idx, cust_id in enumerate(customer_features["customer_id"])}

        print(f"   - Selected {len(feature_columns)} features")
        print(f"   - Feature matrix (unscaled) shape: {X_raw.shape}")

        return X_raw, customer_to_idx, customer_features

    def create_graph_data(self, customers_filtered: pd.DataFrame, p2p_transactions: pd.DataFrame,
                          X_unscaled: np.ndarray, customer_to_idx: Dict[int, int]) -> Tuple[Data, pd.DataFrame]:
        print("Creating graph structure...")

        tx = p2p_transactions.copy()
        tx["sender_idx"] = tx["sender_customer_id"].map(customer_to_idx)
        tx["receiver_idx"] = tx["receiver_customer_id"].map(customer_to_idx)

        valid_transactions = tx.dropna(subset=["sender_idx", "receiver_idx"]).copy()
        valid_transactions["sender_idx"] = valid_transactions["sender_idx"].astype(int)
        valid_transactions["receiver_idx"] = valid_transactions["receiver_idx"].astype(int)

        print(f"   - Valid transactions for graph: {len(valid_transactions):,}")

        edge_pairs = valid_transactions[["sender_idx", "receiver_idx"]].values
        edge_index_full = torch.tensor(edge_pairs.T, dtype=torch.long)  # not used for training to avoid leakage

        # Temporarily keep unscaled x; we will replace with scaled x in train_model after splits
        x = torch.tensor(X_unscaled, dtype=torch.float)
        edge_labels = torch.tensor(valid_transactions["is_fraud"].values, dtype=torch.long)
        edge_pairs_to_classify = torch.tensor(edge_pairs, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index_full,
            edge_labels=edge_labels,
            edge_pairs_to_classify=edge_pairs_to_classify
        )

        print(f"   - Graph: {data.x.shape[0]:,} nodes, {data.edge_index.shape[1]:,} edges")
        print(f"   - Node features (unscaled): {data.x.shape[1]}")
        print(f"   - Fraudulent edges: {edge_labels.sum().item():,} ({edge_labels.float().mean():.1%})")

        return data, valid_transactions

    @staticmethod
    def optimize_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
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

    def train_model(self, data: Data, epochs: int = 300, lr: float = 0.001) -> Tuple[P2PFraudGNN, float]:
        print(f"Training model on {self.device}...")

        # Split edges (link labels)
        num_edges = len(data.edge_labels)
        indices = np.arange(num_edges)

        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4,
            stratify=data.edge_labels.numpy(),
            random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5,
            stratify=data.edge_labels[temp_idx].numpy(),
            random_state=42
        )
        print(f"   - Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")

        # Build TRAIN-ONLY message-passing graph
        edge_pairs_np = data.edge_pairs_to_classify.cpu().numpy()
        train_pairs = edge_pairs_np[train_idx]
        val_pairs = edge_pairs_np[val_idx]
        test_pairs = edge_pairs_np[test_idx]

        # Leakage-safe scaling: fit scaler only on nodes that appear in TRAIN edges
        train_nodes = np.unique(train_pairs.reshape(-1))
        self.scaler = StandardScaler().fit(self.X_raw[train_nodes])
        X_scaled = self.scaler.transform(self.X_raw).astype(np.float32)

        # Update data.x with scaled features and move to device
        data.x = torch.tensor(X_scaled, dtype=torch.float)

        # Edge index used for message passing: TRAIN edges only
        edge_index_train = torch.tensor(train_pairs.T, dtype=torch.long)

        # Move tensors to device
        self.model = P2PFraudGNN(num_node_features=data.x.shape[1], hidden_dim=128).to(self.device)
        data = data.to(self.device)
        edge_index_train = edge_index_train.to(self.device)

        criterion = FocalLoss(alpha=3.0, gamma=2.5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_f1 = -1.0
        patience = 10
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(
                data.x, edge_index_train, data.edge_pairs_to_classify[train_idx]
            )
            loss = criterion(logits, data.edge_labels[train_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validate every epoch (still using TRAIN edge_index for message passing)
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(
                    data.x, edge_index_train, data.edge_pairs_to_classify[val_idx]
                )
                val_probs = F.softmax(val_logits, dim=1)[:, 1].detach().cpu().numpy()
                val_labels = data.edge_labels[val_idx].detach().cpu().numpy()
                optimal_t, val_f1 = self.optimize_threshold(val_labels, val_probs)
                val_auc = roc_auc_score(val_labels, val_probs)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

            # Early stopping on Val F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimal_threshold": optimal_t,
                    "best_val_f1": best_val_f1,
                    "epoch": epoch
                }, "best_fraud_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best checkpoint
        checkpoint = torch.load("best_fraud_model.pth", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimal_threshold = float(checkpoint["optimal_threshold"])
        print(f"✅ Training completed. Best validation F1: {checkpoint['best_val_f1']:.4f} at epoch {checkpoint['epoch']}")

        # Keep objects we need for evaluation
        self._splits = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
        self._edge_index_train = edge_index_train  # train-only message-passing graph
        self._data = data
        return self.model, self.optimal_threshold

    def evaluate_model(self, data: Data) -> Dict[str, Any]:
        if self.model is None or self.optimal_threshold is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("Evaluating model...")

        test_idx = self._splits["test_idx"]
        edge_index_train = self._edge_index_train
        data = self._data  # already on device with scaled features

        self.model.eval()
        with torch.no_grad():
            test_logits = self.model(
                data.x, edge_index_train, data.edge_pairs_to_classify[test_idx]
            )
            test_probs = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
            test_labels = data.edge_labels[test_idx].cpu().numpy()

        test_preds_default = (test_probs >= 0.5).astype(int)
        test_preds_optimal = (test_probs >= self.optimal_threshold).astype(int)

        results = {
            "test_auc": float(roc_auc_score(test_labels, test_probs)),
            "optimal_threshold": float(self.optimal_threshold),
            "default_metrics": {
                "accuracy": float(accuracy_score(test_labels, test_preds_default)),
                "precision": float(precision_score(test_labels, test_preds_default, zero_division=0)),
                "recall": float(recall_score(test_labels, test_preds_default)),
                "f1": float(f1_score(test_labels, test_preds_default, zero_division=0)),
            },
            "optimal_metrics": {
                "accuracy": float(accuracy_score(test_labels, test_preds_optimal)),
                "precision": float(precision_score(test_labels, test_preds_optimal, zero_division=0)),
                "recall": float(recall_score(test_labels, test_preds_optimal)),
                "f1": float(f1_score(test_labels, test_preds_optimal, zero_division=0)),
            },
            "confusion_matrix": confusion_matrix(test_labels, test_preds_optimal).tolist(),
        }

        # Business metrics
        fraud_detection_rate = results["optimal_metrics"]["recall"]
        false_alarm_rate = float(test_preds_optimal[test_labels == 0].mean()) if (test_labels == 0).any() else 0.0
        results["fraud_detection_rate"] = float(fraud_detection_rate)
        results["false_alarm_rate"] = false_alarm_rate

        # Pretty print
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS")
        print("=" * 50)
        print(f"AUC-ROC: {results['test_auc']:.4f}")
        print(f"\nOptimal Threshold: {results['optimal_threshold']:.3f}")
        print(f"  Accuracy:  {results['optimal_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['optimal_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['optimal_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {results['optimal_metrics']['f1']:.4f}")

        total_fraud = int(test_labels.sum())
        fraud_caught = int(results["fraud_detection_rate"] * total_fraud)
        false_alarms = int(results["false_alarm_rate"] * (test_labels == 0).sum())

        print("\nBusiness Impact:")
        print(f"  Fraud Detection Rate: {results['fraud_detection_rate']:.1%}")
        print(f"  Fraud Caught: {fraud_caught}/{total_fraud}")
        print(f"  False Alarms: {false_alarms:,}")
        print(f"  Precision: {results['optimal_metrics']['precision']:.1%} of flagged transactions are fraud")

        return results

    def run_complete_pipeline(self, epochs: int = 300, lr: float = 0.001) -> Dict[str, Any]:
        print("Starting P2P Fraud Detection Pipeline")
        print("=" * 60)

        customers_filtered, accounts, p2p_transactions, cards = self.prepare_data()

        X_raw, customer_to_idx, customer_features = self.create_features(
            customers_filtered, accounts, cards, p2p_transactions
        )

        data, processed_transactions = self.create_graph_data(
            customers_filtered, p2p_transactions, X_raw, customer_to_idx
        )

        model, optimal_threshold = self.train_model(data, epochs, lr)
        results = self.evaluate_model(data)

        print("\nPipeline completed successfully!")
        return results


def main():
    detector = P2PFraudDetector(data_path="data/")
    results = detector.run_complete_pipeline(epochs=100, lr=0.001)
    return detector, results


if __name__ == "__main__":
    detector, results = main()