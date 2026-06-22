"""
HealthRisk AI — Graph Neural Network Module (Day 6)
Builds heterogeneous patient-disease-drug graph and trains
Graph Attention Network (GAT) for outcome prediction.

Architecture:
  - Node types: Patient, Disease (ICD-10), Drug (NDC/ATC)
  - Edge types: has_diagnosis, prescribed, drug_interaction, comorbidity_link
  - Model: 3-layer GAT with 4 attention heads + residual connections
  - Tasks: ICU mortality, 30-day readmission, ICU transfer

Targets: AUROC > 0.78 (mortality), AUROC > 0.72 (readmission)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict

# PyTorch Geometric imports (conditional)
try:
    from torch_geometric.data import HeteroData, Data
    from torch_geometric.nn import GATConv, SAGEConv, to_hetero
    import torch_geometric.transforms as T
    HAS_PYG = True
except ImportError:
    print("Warning: torch-geometric not installed. Using simplified GNN.")
    HAS_PYG = False


# ──────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────

class HealthRiskGraphBuilder:
    """
    Constructs a heterogeneous patient-disease-drug graph from MIMIC-IV data.
    """

    def __init__(self):
        self.patient_ids = []
        self.disease_ids = []
        self.drug_ids = []
        self.patient_features = None
        self.disease_features = None
        self.drug_features = None

    def build_patient_nodes(self, features_df: pd.DataFrame) -> np.ndarray:
        """Build patient node feature matrix."""
        feature_cols = [
            "age", "charlson_index", "num_prior_admissions",
            "Creatinine_last", "Hemoglobin_last", "WBC_last",
            "Sodium_last", "Lactate_last", "n_medications",
            "los_days", "icu_flag", "gender_binary",
        ]
        available = [c for c in feature_cols if c in features_df.columns]
        X = features_df[available].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.patient_ids = features_df["hadm_id"].tolist()
        self.patient_features = X_scaled
        return X_scaled

    def build_disease_nodes(self, diagnoses_df: pd.DataFrame) -> np.ndarray:
        """Build disease node features using ICD code one-hot encoding."""
        unique_codes = diagnoses_df["icd_code"].unique()
        self.disease_ids = list(unique_codes)
        n_diseases = len(unique_codes)
        # Simple positional encoding for ICD codes
        features = np.eye(min(n_diseases, 64))[:n_diseases]
        if n_diseases < 64:
            pad = np.zeros((n_diseases, 64 - n_diseases))
            features = np.hstack([features, pad])
        self.disease_features = features
        return features

    def build_drug_nodes(self, prescriptions_df: pd.DataFrame) -> np.ndarray:
        """Build drug node features."""
        if "drug" not in prescriptions_df.columns:
            self.drug_ids = []
            self.drug_features = np.zeros((0, 32))
            return self.drug_features
        unique_drugs = prescriptions_df["drug"].unique()
        self.drug_ids = list(unique_drugs)
        n_drugs = len(unique_drugs)
        features = np.random.randn(n_drugs, 32) * 0.1  # Placeholder molecular features
        self.drug_features = features
        return features

    def build_edges(
        self,
        diagnoses_df: pd.DataFrame,
        prescriptions_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Build edge index tensors for each edge type."""
        edges = {}

        # patient → disease edges (has_diagnosis)
        patient_map = {pid: i for i, pid in enumerate(self.patient_ids)}
        disease_map = {did: i for i, did in enumerate(self.disease_ids)}

        p2d_src, p2d_dst = [], []
        for _, row in diagnoses_df.iterrows():
            if row["hadm_id"] in patient_map and row["icd_code"] in disease_map:
                p2d_src.append(patient_map[row["hadm_id"]])
                p2d_dst.append(disease_map[row["icd_code"]])
        edges["patient_disease"] = (np.array(p2d_src), np.array(p2d_dst))

        # patient → drug edges (prescribed)
        if prescriptions_df is not None and "drug" in prescriptions_df.columns:
            drug_map = {did: i for i, did in enumerate(self.drug_ids)}
            p2rx_src, p2rx_dst = [], []
            for _, row in prescriptions_df.iterrows():
                if row["hadm_id"] in patient_map and row["drug"] in drug_map:
                    p2rx_src.append(patient_map[row["hadm_id"]])
                    p2rx_dst.append(drug_map[row["drug"]])
            edges["patient_drug"] = (np.array(p2rx_src), np.array(p2rx_dst))

        return edges

    def to_pyg_data(
        self,
        features_df: pd.DataFrame,
        diagnoses_df: pd.DataFrame,
        labels: np.ndarray,
        prescriptions_df: Optional[pd.DataFrame] = None,
    ):
        """Build PyG HeteroData object."""
        # Build node features
        patient_x = torch.tensor(
            self.build_patient_nodes(features_df), dtype=torch.float32
        )
        disease_x = torch.tensor(
            self.build_disease_nodes(diagnoses_df), dtype=torch.float32
        )

        edges = self.build_edges(diagnoses_df, prescriptions_df)

        if HAS_PYG:
            data = HeteroData()
            data["patient"].x = patient_x
            data["patient"].y = torch.tensor(labels, dtype=torch.long)
            data["disease"].x = disease_x

            if "patient_disease" in edges and len(edges["patient_disease"][0]) > 0:
                src, dst = edges["patient_disease"]
                data["patient", "has_diagnosis", "disease"].edge_index = torch.tensor(
                    np.vstack([src, dst]), dtype=torch.long
                )
                # Reverse edges
                data["disease", "rev_has_diagnosis", "patient"].edge_index = torch.tensor(
                    np.vstack([dst, src]), dtype=torch.long
                )
            return data
        else:
            # Fallback: homogeneous graph (patient nodes only)
            data_simple = SimpleGraphData(patient_x, torch.tensor(labels, dtype=torch.long))
            return data_simple


class SimpleGraphData:
    """Fallback when PyG is not available."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


# ──────────────────────────────────────────────────────
# GRAPH ATTENTION NETWORK
# ──────────────────────────────────────────────────────

class HealthRiskGAT(nn.Module):
    """
    3-layer Graph Attention Network for patient outcome prediction.
    4 attention heads per layer, residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout

        if HAS_PYG:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
            # Residual projection
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            # Simple MLP fallback
            self.conv1 = nn.Linear(in_channels, hidden_channels)
            self.conv2 = nn.Linear(hidden_channels, hidden_channels)
            self.conv3 = nn.Linear(hidden_channels, out_channels)
            self.residual = nn.Linear(in_channels, out_channels)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, edge_index=None):
        residual = self.residual(x)

        if HAS_PYG and edge_index is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv2(x))
            x = self.conv3(x)

        x = x + residual  # Residual connection
        return self.classifier(x)

    def get_embeddings(self, x, edge_index=None):
        """Return node embeddings (before classifier)."""
        residual = self.residual(x)
        if HAS_PYG and edge_index is not None:
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
        else:
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = self.conv3(x)
        return x + residual


# ──────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ──────────────────────────────────────────────────────

class GNNTrainer:
    """Handles GAT model training, evaluation, and embedding extraction."""

    def __init__(self, model: HealthRiskGAT, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = {"train_loss": [], "val_auroc": []}

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_size: float = 0.20,
    ) -> Dict[str, float]:
        """Train the GAT model."""
        x, y = x.to(self.device), y.to(self.device)
        n = x.size(0)
        indices = torch.randperm(n)
        split = int(n * (1 - val_size))
        train_idx, val_idx = indices[:split], indices[split:]

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_auroc = 0.0
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = criterion(out[train_idx], y[train_idx])
            loss.backward()
            optimizer.step()

            # Validation every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(x, edge_index)
                probs = torch.softmax(val_out[val_idx], dim=-1)[:, 1].cpu().numpy()
                y_val = y[val_idx].cpu().numpy()
                try:
                    auroc = roc_auc_score(y_val, probs)
                except Exception:
                    auroc = 0.5
                self.history["val_auroc"].append(auroc)
                self.history["train_loss"].append(loss.item())
                print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | Val AUROC: {auroc:.4f}")
                if auroc > best_auroc:
                    best_auroc = auroc

        print(f"\nBest Val AUROC: {best_auroc:.4f}")
        return {"best_val_auroc": best_auroc}

    def get_patient_embeddings(
        self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Extract patient embeddings for use in ensemble."""
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_embeddings(x, edge_index)
        return embeddings.cpu().numpy()


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing HealthRiskGAT with synthetic data...")
    n_patients = 1000
    in_channels = 12

    x = torch.randn(n_patients, in_channels)
    y = torch.randint(0, 2, (n_patients,))

    model = HealthRiskGAT(in_channels=in_channels, hidden_channels=64, out_channels=32)
    trainer = GNNTrainer(model, device="cpu")
    results = trainer.train(x, y, edge_index=None, epochs=30, lr=1e-3)
    print(f"Results: {results}")

    embeddings = trainer.get_patient_embeddings(x)
    print(f"Patient embeddings shape: {embeddings.shape}")
