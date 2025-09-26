"""
P2P Fraud Detection using Graph Neural Networks

This module implements a fraud detection system for peer-to-peer transactions
using Graph Neural Networks with attention mechanisms. The system is designed
to handle imbalanced datasets

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
    Focal Loss implementation for handling class imbalanced datasets.

    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing learning on hard negatives.

    Args:
        alpha (float): Weighting factor for rare class (fraud). Default: 2.0
        gamma (float): Focusing parameter. Higher gamma puts more focus on hard examples. Default: 2.0
    """

    def __init__(self, alpha: float = 2.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class P2PFraudGNN(nn.Module):
    """
    Graph Neural Network for P2P fraud detection.

    Architecture:
    - Graph Attention Network (GAT) layers for learning node representations
    - Graph Convolution Network (GCN) layer for final node embeddings
    - MLP classifier for edge (transaction) classification

    Args:
        num_node_features (int): Number of input node features
        hidden_dim (int): Hidden dimension size. Default: 128
        num_layers (int): Number of GNN layers. Default: 3
    """

    def __init__(self, num_node_features: int, hidden_dim: int = 128, num_layers: int = 3):
        super(P2PFraudGNN, self).__init__()

        # Graph Neural Network layers
        self.conv1 = GATConv(
            num_node_features,
            hidden_dim // 4,
            heads=4,
            dropout=0.3,
            concat=True
        )
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim // 2,
            heads=2,
            dropout=0.3,
            concat=True
        )
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Edge classifier: takes concatenated source and target node embeddings
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

            nn.Linear(hidden_dim // 4, 2)  # Binary classification: fraud or legitimate
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_pairs_to_classify: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_pairs_to_classify: Edge pairs to classify [num_p2p_edges, 2]

        Returns:
            Edge predictions [num_p2p_edges, 2]
        """
        # First GAT layer with multi-head attention
        x1 = F.elu(self.conv1(x, edge_index))
        x1 = self.bn1(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)

        # Second GAT layer
        x2 = F.elu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)
        x2 = F.dropout(x2, p=0.3, training=self.training)

        # Final GCN layer
        x3 = F.elu(self.conv3(x2, edge_index))
        x3 = self.bn3(x3)

        # Residual connection if dimensions match
        if x3.size(-1) == x1.size(-1):
            x3 = x3 + x1

        # Extract embeddings for edge endpoints
        src_embeddings = x3[edge_pairs_to_classify[:, 0]]  # Source node embeddings
        dst_embeddings = x3[edge_pairs_to_classify[:, 1]]  # Destination node embeddings

        # Concatenate source and destination embeddings for edge classification
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        # Classify edge as fraud or legitimate
        edge_predictions = self.edge_classifier(edge_embeddings)

        return edge_predictions


class P2PFraudDetector:
    """
    Main class for P2P fraud detection system.

    This class handles the entire pipeline from data loading to model training
    and evaluation, ensuring no data leakage occurs.
    """

    def __init__(self, data_path: str = "data/"):
        """
        Initialize the fraud detector.

        Args:
            data_path: Path to directory containing CSV files
        """
        self.data_path = data_path
        self.model = None
        self.optimal_threshold = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required CSV files.

        Returns:
            Tuple of (customers, accounts, p2p_transactions, cards) DataFrames
        """
        print("Loading data...")

        try:
            customers = pd.read_csv(f"{self.data_path}customers.csv")
            accounts = pd.read_csv(f"{self.data_path}accounts.csv")
            p2p_transactions = pd.read_csv(f"{self.data_path}p2p_transactions.csv")
            cards = pd.read_csv(f"{self.data_path}cards.csv")

            print(f"✅ Data loaded successfully:")
            print(f"   - Customers: {len(customers):,}")
            print(f"   - P2P transactions: {len(p2p_transactions):,}")
            print(f"   - Fraud rate: {p2p_transactions['is_fraud'].mean():.1%}")

            return customers, accounts, p2p_transactions, cards

        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            raise

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data by mapping P2P transactions to customers.

        Returns:
            Tuple of filtered and mapped DataFrames
        """
        customers, accounts, p2p_transactions, cards = self.load_data()

        # Create customer-account mapping
        customer_account_map = dict(zip(accounts["account_id"], accounts["customer_id"]))

        # Map P2P transactions to customers (both sender and receiver)
        p2p_transactions["sender_customer_id"] = (
            p2p_transactions["sender_account_id"].map(customer_account_map)
        )
        p2p_transactions["receiver_customer_id"] = (
            p2p_transactions["receiver_account_id"].map(customer_account_map)
        )

        # Remove transactions where customer mapping failed
        initial_count = len(p2p_transactions)
        p2p_transactions = p2p_transactions.dropna(
            subset=["sender_customer_id", "receiver_customer_id"]
        )
        print(f"   - P2P transactions after mapping: {len(p2p_transactions):,} "
              f"({initial_count - len(p2p_transactions):,} dropped)")

        # Filter customers to only those involved in P2P transactions
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
        """
        Create node features

        Args:
            customers: Customer information
            accounts: Account information
            cards: Card information
            p2p_transactions: P2P transaction data

        Returns:
            Tuple of (feature_matrix, customer_to_index_mapping, customer_features_df)
        """
        print("Creating clean features...")

        customer_features = customers.copy()

        # Encode categorical customer attributes
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

        # Account statistics
        account_stats = accounts.groupby("customer_id").agg({
            "is_active": "sum",  # Number of active accounts
            "account_id": "count"  # Total number of accounts
        }).rename(columns={"account_id": "num_accounts"})

        # Transaction behavior features
        # Sender behavior patterns
        sender_stats = p2p_transactions.groupby("sender_customer_id").agg({
            "amount": ["count", "mean", "std", "max", "min"],  # Transaction patterns
            "receiver_customer_id": "nunique"  # Number of unique recipients
        })
        sender_stats.columns = ["_".join(col).strip() for col in sender_stats.columns]
        sender_stats = sender_stats.add_prefix("sender_")

        # Receiver behavior patterns
        receiver_stats = p2p_transactions.groupby("receiver_customer_id").agg({
            "amount": ["count", "mean", "std", "max", "min"],  # Received transaction patterns
            "sender_customer_id": "nunique"  # Number of unique senders
        })
        receiver_stats.columns = ["_".join(col).strip() for col in receiver_stats.columns]
        receiver_stats = receiver_stats.add_prefix("receiver_")

        # Temporal patterns (if timestamp is available)
        temporal_stats = pd.DataFrame()
        if "timestamp" in p2p_transactions.columns:
            p2p_transactions["timestamp"] = pd.to_datetime(p2p_transactions["timestamp"])
            p2p_transactions["hour"] = p2p_transactions["timestamp"].dt.hour
            p2p_transactions["day_of_week"] = p2p_transactions["timestamp"].dt.dayofweek

            temporal_stats = p2p_transactions.groupby("sender_customer_id").agg({
                "hour": ["mean", "std"],  # Time-of-day patterns
                "day_of_week": ["mean", "std"]  # Day-of-week patterns
            })
            temporal_stats.columns = ["_".join(col).strip() for col in temporal_stats.columns]
            temporal_stats = temporal_stats.add_prefix("temporal_")

        # Card information
        card_stats = pd.DataFrame(columns=["num_cards", "num_active_cards"])
        if len(cards) > 0:
            card_stats = cards.groupby("customer_id").agg({
                "is_active": "sum",  # Number of active cards
                "card_id": "count"  # Total number of cards
            }).rename(columns={"card_id": "num_cards", "is_active": "num_active_cards"})

        # Merge all feature sets
        feature_dfs = [account_stats, sender_stats, receiver_stats, card_stats]
        if not temporal_stats.empty:
            feature_dfs.append(temporal_stats)

        for df in feature_dfs:
            customer_features = customer_features.merge(
                df, left_on="customer_id", right_index=True, how="left"
            )

        # Fill missing values with zeros
        customer_features = customer_features.fillna(0)

        # Select only numerical features for the model
        feature_columns = [
            col for col in customer_features.columns
            if (col not in ["customer_id"] and
                customer_features[col].dtype in ["int64", "float64", "int32", "float32"])
        ]

        print(f"   - Selected {len(feature_columns)} clean features")
        print(f"   - Features: {feature_columns[:5]}..." if len(
            feature_columns) > 5 else f"   - Features: {feature_columns}")

        # Create feature matrix
        X = customer_features[feature_columns].values.astype(np.float32)

        # Standardize features for better training stability
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Create mapping from customer_id to matrix index
        customer_to_idx = {
            cust_id: idx for idx, cust_id in enumerate(customer_features["customer_id"])
        }

        print(f"   - Feature matrix shape: {X.shape}")

        return X, customer_to_idx, customer_features

    def create_graph_data(self, customers_filtered: pd.DataFrame, p2p_transactions: pd.DataFrame,
                          X: np.ndarray, customer_to_idx: Dict[int, int]) -> Tuple[Data, pd.DataFrame]:
        """
        Create PyTorch Geometric Data object for the graph.

        Args:
            customers_filtered: Filtered customer data
            p2p_transactions: P2P transaction data
            X: Node feature matrix
            customer_to_idx: Mapping from customer_id to matrix index

        Returns:
            Tuple of (graph_data, processed_transactions)
        """
        print("Creating graph structure...")

        # Map transaction customer IDs to matrix indices
        p2p_transactions["sender_idx"] = p2p_transactions["sender_customer_id"].map(customer_to_idx)
        p2p_transactions["receiver_idx"] = p2p_transactions["receiver_customer_id"].map(customer_to_idx)

        # Remove transactions where mapping failed
        valid_transactions = p2p_transactions.dropna(subset=["sender_idx", "receiver_idx"])
        valid_transactions["sender_idx"] = valid_transactions["sender_idx"].astype(int)
        valid_transactions["receiver_idx"] = valid_transactions["receiver_idx"].astype(int)

        print(f"   - Valid transactions for graph: {len(valid_transactions):,}")

        # Create edge structure: [2, num_edges] tensor
        edge_pairs = valid_transactions[["sender_idx", "receiver_idx"]].values
        edge_index = torch.tensor(edge_pairs.T, dtype=torch.long)

        # Node features
        x = torch.tensor(X, dtype=torch.float)

        # Edge labels (fraud/legitimate)
        edge_labels = torch.tensor(valid_transactions["is_fraud"].values, dtype=torch.long)

        # Edge pairs to classify (same as edge_index but different format for model)
        edge_pairs_to_classify = torch.tensor(edge_pairs, dtype=torch.long)

        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_labels=edge_labels,
            edge_pairs_to_classify=edge_pairs_to_classify
        )

        print(f"   - Graph: {data.x.shape[0]:,} nodes, {data.edge_index.shape[1]:,} edges")
        print(f"   - Node features: {data.x.shape[1]}")
        print(f"   - Fraudulent edges: {edge_labels.sum().item():,} ({edge_labels.float().mean():.1%})")

        return data, valid_transactions

    def optimize_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
        """
        Find optimal classification threshold by maximizing F1-score.

        For imbalanced datasets, the default threshold of 0.5 is often suboptimal.
        This function finds the threshold that maximizes F1-score on validation data.

        Args:
            y_true: True labels
            y_probs: Predicted probabilities

        Returns:
            Tuple of (optimal_threshold, best_f1_score)
        """
        thresholds = np.arange(0.001, 0.999, 0.001)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)

            # Skip if all predictions are the same class
            if len(np.unique(y_pred)) == 1:
                f1_scores.append(0)
                continue

            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        best_f1 = f1_scores[optimal_idx]

        return optimal_threshold, best_f1

    def train_model(self, data: Data, epochs: int = 300, lr: float = 0.001) -> Tuple[P2PFraudGNN, float]:
        """
        Train the fraud detection model.

        Args:
            data: Graph data object
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Tuple of (trained_model, optimal_threshold)
        """
        print(f"Training model on {self.device}...")

        # Create stratified train/validation/test splits
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

        # Initialize model
        self.model = P2PFraudGNN(num_node_features=data.x.shape[1], hidden_dim=128)
        self.model.to(self.device)
        data = data.to(self.device)

        # Loss function optimized for class imbalance
        criterion = FocalLoss(alpha=3.0, gamma=2.5)

        # Optimizer with weight decay for regularization
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Training loop with early stopping
        best_val_f1 = 0
        patience_counter = 0
        patience = 5

        for epoch in range(epochs):
            # Training step
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            out = self.model(data.x, data.edge_index, data.edge_pairs_to_classify[train_idx])
            loss = criterion(out, data.edge_labels[train_idx])

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation every 20 epochs
            if epoch % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(data.x, data.edge_index, data.edge_pairs_to_classify[val_idx])
                    val_probs = F.softmax(val_out, dim=1)[:, 1].cpu().numpy()
                    val_labels = data.edge_labels[val_idx].cpu().numpy()

                    # Find optimal threshold and compute metrics
                    optimal_threshold, val_f1 = self.optimize_threshold(val_labels, val_probs)
                    val_auc = roc_auc_score(val_labels, val_probs)

                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

                    # Save best model
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        torch.save({
                            "model_state_dict": self.model.state_dict(),
                            "optimal_threshold": optimal_threshold,
                            "best_val_f1": best_val_f1,
                            "epoch": epoch
                        }, "best_fraud_model.pth")
                    else:
                        patience_counter += 1

                    # Early stopping
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        # Load best model
        checkpoint = torch.load("best_fraud_model.pth", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimal_threshold = checkpoint["optimal_threshold"]

        print(f"✅ Training completed. Best validation F1: {checkpoint['best_val_f1']:.4f}")

        return self.model, self.optimal_threshold

    def evaluate_model(self, data: Data) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Args:
            data: Graph data object

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None or self.optimal_threshold is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("Evaluating model...")

        # Recreate test split (same random seed ensures consistency)
        num_edges = len(data.edge_labels)
        indices = np.arange(num_edges)

        _, temp_idx = train_test_split(
            indices, test_size=0.4,
            stratify=data.edge_labels.numpy(),
            random_state=42
        )
        _, test_idx = train_test_split(
            temp_idx, test_size=0.5,
            stratify=data.edge_labels[temp_idx].numpy(),
            random_state=42
        )

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            test_out = self.model(data.x, data.edge_index, data.edge_pairs_to_classify[test_idx])
            test_probs = F.softmax(test_out, dim=1)[:, 1].cpu().numpy()
            test_labels = data.edge_labels[test_idx].cpu().numpy()

            # Predictions with different thresholds
            test_preds_default = (test_probs >= 0.5).astype(int)
            test_preds_optimal = (test_probs >= self.optimal_threshold).astype(int)

            # Compute comprehensive metrics
            results = {
                "test_auc": roc_auc_score(test_labels, test_probs),
                "optimal_threshold": self.optimal_threshold,

                # Default threshold metrics
                "default_metrics": {
                    "accuracy": accuracy_score(test_labels, test_preds_default),
                    "precision": precision_score(test_labels, test_preds_default, zero_division=0),
                    "recall": recall_score(test_labels, test_preds_default),
                    "f1": f1_score(test_labels, test_preds_default, zero_division=0)
                },

                # Optimal threshold metrics
                "optimal_metrics": {
                    "accuracy": accuracy_score(test_labels, test_preds_optimal),
                    "precision": precision_score(test_labels, test_preds_optimal, zero_division=0),
                    "recall": recall_score(test_labels, test_preds_optimal),
                    "f1": f1_score(test_labels, test_preds_optimal, zero_division=0)
                },

                # Business metrics
                "confusion_matrix": confusion_matrix(test_labels, test_preds_optimal).tolist(),
                "fraud_detection_rate": recall_score(test_labels, test_preds_optimal),
                "false_alarm_rate": (test_preds_optimal[test_labels == 0]).mean(),
            }

        # Print results
        print(f"\n{'=' * 50}")
        print(f"📊 TEST RESULTS")
        print(f"{'=' * 50}")
        print(f"AUC-ROC: {results['test_auc']:.4f}")
        print(f"\nOptimal Threshold: {results['optimal_threshold']:.3f}")
        print(f"  Accuracy:  {results['optimal_metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['optimal_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['optimal_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {results['optimal_metrics']['f1']:.4f}")

        print(f"\nBusiness Impact:")
        fraud_caught = int(results["fraud_detection_rate"] * test_labels.sum())
        total_fraud = int(test_labels.sum())
        false_alarms = int(results["false_alarm_rate"] * (test_labels == 0).sum())

        print(f"  Fraud Detection Rate: {results['fraud_detection_rate']:.1%}")
        print(f"  Fraud Caught: {fraud_caught}/{total_fraud}")
        print(f"  False Alarms: {false_alarms:,}")
        print(f"  Precision: {results['optimal_metrics']['precision']:.1%} of flagged transactions are fraud")

        return results

    def run_complete_pipeline(self, epochs: int = 300, lr: float = 0.001) -> Dict[str, Any]:
        """
        Run the complete fraud detection pipeline.

        Args:
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Dictionary containing evaluation results
        """
        print("Starting P2P Fraud Detection Pipeline")
        print("=" * 60)

        # Step 1: Load and prepare data
        customers_filtered, accounts, p2p_transactions, cards = self.prepare_data()

        # Step 2: Create features
        X, customer_to_idx, customer_features = self.create_features(
            customers_filtered, accounts, cards, p2p_transactions
        )

        # Step 3: Create graph structure
        data, processed_transactions = self.create_graph_data(
            customers_filtered, p2p_transactions, X, customer_to_idx
        )

        # Step 4: Train model
        model, optimal_threshold = self.train_model(data, epochs, lr)

        # Step 5: Evaluate model
        results = self.evaluate_model(data)

        print(f"\n Pipeline completed successfully!")
        return results


def main():
    """
    Main function to run the fraud detection system.
    """
    # Initialize fraud detector
    detector = P2PFraudDetector(data_path="data/")

    # Run complete pipeline
    results = detector.run_complete_pipeline(epochs=300, lr=0.001)

    return detector, results


if __name__ == "__main__":
    detector, results = main()