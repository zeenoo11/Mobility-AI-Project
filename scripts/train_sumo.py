"""
Training Pipeline for EV Charging Demand Prediction (Graph WaveNet)

End-to-end: data loading → model training → evaluation → visualization.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from src.models.graph_wavenet import GraphWaveNet, count_parameters
from src.sumo_dataset import load_dataset
from src.train import LinearWarmupCosineScheduler


def train_epoch(model, loader, optimizer, criterion, adj, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x, adj)  # (B, N, T_pred, 1)
        pred = pred.squeeze(-1)  # (B, N, T_pred)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def eval_epoch(model, loader, criterion, adj, device):
    model.eval()
    losses = []
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, adj).squeeze(-1)
        loss = criterion(pred, y)
        losses.append(loss.item())
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Metrics
    mse = nn.functional.mse_loss(preds, targets).item()
    mae = (preds - targets).abs().mean().item()

    # R² (per station, then average)
    ss_res = ((targets - preds) ** 2).sum(dim=(0, 2))  # (N,)
    ss_tot = ((targets - targets.mean(dim=(0, 2), keepdim=True)) ** 2).sum(dim=(0, 2))
    r2_per_station = 1 - ss_res / (ss_tot + 1e-8)
    r2_mean = r2_per_station.mean().item()

    return {
        "loss": np.mean(losses),
        "rmse": np.sqrt(mse),
        "mae": mae,
        "r2": r2_mean,
    }


def train(config: dict | None = None):
    """Main training function."""
    if config is None:
        config = {
            "t_input": 12,
            "t_pred": 1,
            "batch_size": 16,
            "hidden_dim": 32,
            "n_blocks": 2,
            "n_layers_per_block": 3,
            "node_emb_dim": 16,
            "graph_order": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "epochs": 100,
            "patience": 15,
            "warmup_epochs": 5,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir = Path(PROJECT_ROOT) / "data" / "sumo" / "sim_outputs"
    data = load_dataset(
        data_dir,
        t_input=config["t_input"],
        t_pred=config["t_pred"],
        batch_size=config["batch_size"],
    )
    adj = data["adj"].to(device)
    print(f"Dataset: {json.dumps(data['info'], indent=2)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = GraphWaveNet(
        n_nodes=data["n_stations"],
        in_features=data["n_features"],
        out_features=1,
        hidden_dim=config["hidden_dim"],
        n_blocks=config["n_blocks"],
        n_layers_per_block=config["n_layers_per_block"],
        node_emb_dim=config["node_emb_dim"],
        graph_order=config["graph_order"],
        dropout=config["dropout"],
        pred_horizon=config["t_pred"],
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-4
    )
    T_0 = max(config["epochs"] - config["warmup_epochs"], 10)
    scheduler = LinearWarmupCosineScheduler(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        T_0=T_0,
    )
    criterion = nn.MSELoss()

    # ── Training Loop ───────────────────────────────────────────────────────
    results_dir = Path(PROJECT_ROOT) / "results" / "sumo_demand"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "best_model.pt"

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_rmse": [],
               "val_mae": [], "val_r2": [], "lr": []}

    t0 = time.time()
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch(
            model, data["train"], optimizer, criterion, adj, device
        )
        val_metrics = eval_epoch(model, data["val"], criterion, adj, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])
        history["lr"].append(lr)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_metrics['loss']:.6f} | "
                  f"RMSE: {val_metrics['rmse']:.4f} | "
                  f"R²: {val_metrics['r2']:.4f} | "
                  f"LR: {lr:.2e} | {elapsed:.1f}s")

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stop at epoch {epoch}")
                break

    # ── Test Evaluation ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = eval_epoch(model, data["test"], criterion, adj, device)

    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"{'='*50}")

    # ── Save Results ────────────────────────────────────────────────────────
    results = {
        "config": config,
        "n_params": n_params,
        "test_metrics": test_metrics,
        "history": history,
        "best_val_loss": best_val_loss,
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save learned adjacency
    adj_learned = model.get_learned_adjacency()
    if adj_learned is not None:
        np.save(results_dir / "learned_adj.npy", adj_learned.cpu().numpy())

    print(f"Results saved to {results_dir}")
    return results


if __name__ == "__main__":
    train()
