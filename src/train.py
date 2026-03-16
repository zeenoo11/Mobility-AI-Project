"""
Training utilities for EV energy prediction models.
Supports early stopping, AdamW optimizer, MSELoss.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader,
                epochs=30, lr=1e-3, patience=5,
                checkpoint_dir='results/checkpoints',
                model_name='model', device='cpu', verbose=True):
    """
    Train a model with AdamW + MSELoss + early stopping.

    Returns:
        history: dict with 'train_loss' and 'val_loss' lists
        best_val_loss: float
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_losses = []
        for drv, bms, tgt in train_loader:
            drv, bms, tgt = drv.to(device), bms.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(drv, bms)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for drv, bms, tgt in val_loader:
                drv, bms, tgt = drv.to(device), bms.to(device), tgt.to(device)
                pred = model(drv, bms)
                loss = criterion(pred, tgt)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"  [{model_name}] Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  [{model_name}] Early stop at epoch {epoch} "
                          f"(best val loss: {best_val_loss:.4f})")
                break

    # Load best checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if verbose:
        print(f"  [{model_name}] Training done. Best val loss: {best_val_loss:.4f}")

    return history, best_val_loss
