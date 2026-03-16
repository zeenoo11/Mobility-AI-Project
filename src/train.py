"""
Training utilities for EV energy prediction models.
Supports early stopping, AdamW optimizer, MSELoss,
CosineAnnealingWarmRestarts with linear warmup.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn


class LinearWarmupCosineScheduler:
    """Linear warmup followed by CosineAnnealingWarmRestarts."""

    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=1, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        self._step_count = 0

    def step(self, epoch=None):
        self._step_count += 1
        if self._step_count <= self.warmup_epochs:
            scale = self._step_count / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * scale
        else:
            self.cosine.step()

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def train_model(model, train_loader, val_loader,
                epochs=100, lr=1e-3, patience=15,
                warmup_epochs=5, scheduler_type='cosine_warmup',
                checkpoint_dir='results/checkpoints',
                model_name='model', device='cpu', verbose=True):
    """
    Train a model with AdamW + MSELoss + early stopping.

    Args:
        scheduler_type: 'cosine_warmup' (default) or 'plateau' (legacy)
        warmup_epochs: number of linear warmup epochs (only for cosine_warmup)

    Returns:
        history: dict with 'train_loss', 'val_loss', 'lr' lists
        best_val_loss: float
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if scheduler_type == 'cosine_warmup':
        T_0 = max(epochs - warmup_epochs, 10)
        scheduler = LinearWarmupCosineScheduler(
            optimizer, warmup_epochs=warmup_epochs,
            T_0=T_0, T_mult=1, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

    criterion = nn.MSELoss()

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

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
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Step scheduler
        if scheduler_type == 'cosine_warmup':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"  [{model_name}] Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

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
                          f"(best val loss: {best_val_loss:.6f})")
                break

    # Load best checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if verbose:
        total_epochs = len(history['train_loss'])
        print(f"  [{model_name}] Training done. {total_epochs} epochs, "
              f"best val loss: {best_val_loss:.6f}")

    return history, best_val_loss
