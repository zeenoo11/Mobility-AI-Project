"""
PyTorch Dataset for SUMO EV Charging Demand Prediction

Loads station features from parsed simulation outputs and creates
input-target pairs for temporal prediction.

Input:  (N_stations, T_input, F) features from past time bins
Target: (N_stations, T_pred, 1) demand in next time bins
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ChargingDemandDataset(Dataset):
    """
    Dataset for charging demand prediction.

    Each sample:
      - x: (N, T_in, F) station features for T_in past time steps
      - y: (N, T_pred) charging demand for T_pred future steps

    Uses sliding window over temporal axis within each simulation run.
    """

    def __init__(self, features: np.ndarray, t_input: int = 12,
                 t_pred: int = 1, target_idx: int = 0):
        """
        Args:
            features: (N_runs, N_stations, N_bins, N_features)
            t_input: number of input time steps
            t_pred: number of prediction time steps
            target_idx: feature index to predict (0=num_arrivals)
        """
        self.t_input = t_input
        self.t_pred = t_pred
        self.target_idx = target_idx

        # Build index: (run_idx, start_t)
        self.samples = []
        n_runs, n_stations, n_bins, n_features = features.shape

        for run_idx in range(n_runs):
            max_start = n_bins - t_input - t_pred
            for start_t in range(max_start + 1):
                self.samples.append((run_idx, start_t))

        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        run_idx, start_t = self.samples[idx]
        end_input = start_t + self.t_input
        end_pred = end_input + self.t_pred

        x = self.features[run_idx, :, start_t:end_input, :]
        y = self.features[run_idx, :, end_input:end_pred, self.target_idx]

        return x, y  # x: (N, T_in, F), y: (N, T_pred)


def load_dataset(data_dir: Path, t_input: int = 12, t_pred: int = 1,
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 batch_size: int = 16) -> dict:
    """
    Load and split dataset into train/val/test DataLoaders.

    Args:
        data_dir: directory containing station_features.npy
        t_input: input window length
        t_pred: prediction horizon
        train_ratio: fraction for training
        val_ratio: fraction for validation
        batch_size: batch size for DataLoaders

    Returns:
        dict with 'train', 'val', 'test' DataLoaders,
        'adj' adjacency matrix, and dataset info
    """
    features = np.load(data_dir / "station_features.npy")
    n_runs = features.shape[0]

    # Normalize features (per-feature min-max across all runs)
    feat_shape = features.shape
    feat_flat = features.reshape(-1, feat_shape[-1])
    feat_min = feat_flat.min(axis=0, keepdims=True)
    feat_max = feat_flat.max(axis=0, keepdims=True)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    features_norm = (features - feat_min.reshape(1, 1, 1, -1)) / \
                     feat_range.reshape(1, 1, 1, -1)

    # Split by runs
    indices = np.random.RandomState(42).permutation(n_runs)
    n_train = int(n_runs * train_ratio)
    n_val = int(n_runs * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = ChargingDemandDataset(features_norm[train_idx], t_input, t_pred)
    val_ds = ChargingDemandDataset(features_norm[val_idx], t_input, t_pred)
    test_ds = ChargingDemandDataset(features_norm[test_idx], t_input, t_pred)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load adjacency matrix
    graph_dir = data_dir.parent / "graph_data"
    adj = np.load(graph_dir / "adj.npy")
    adj_tensor = torch.tensor(adj, dtype=torch.float32)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "adj": adj_tensor,
        "n_stations": features.shape[1],
        "n_features": features.shape[3],
        "feat_min": feat_min,
        "feat_max": feat_max,
        "info": {
            "n_runs": n_runs,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "t_input": t_input,
            "t_pred": t_pred,
        },
    }


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "sumo" / "sim_outputs"
    if not (data_dir / "station_features.npy").exists():
        print("No dataset found. Run parse_outputs.py --all first.")
        print("Creating dummy dataset for testing...")

        # Create dummy data: 10 runs
        dummy = np.random.rand(10, 53, 24, 6).astype(np.float32)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / "station_features.npy", dummy)

    data = load_dataset(data_dir, t_input=12, t_pred=1, batch_size=4)
    print(f"Dataset info: {json.dumps(data['info'], indent=2)}")

    # Test one batch
    for x, y in data["train"]:
        print(f"Batch x shape: {x.shape}")  # (B, N, T_in, F)
        print(f"Batch y shape: {y.shape}")  # (B, N, T_pred)
        print(f"Adj shape:     {data['adj'].shape}")
        break
