"""
Evaluation utilities: RMSE, MAE, R², attention weight extraction.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, loader, tgt_scaler, device='cpu'):
    """
    Run inference on a DataLoader and compute metrics in original scale.

    Returns:
        metrics: dict with 'rmse', 'mae', 'r2'
        preds:   numpy array of predictions (original scale)
        actuals: numpy array of ground truth (original scale)
    """
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for drv, bms, tgt in loader:
            drv, bms, tgt = drv.to(device), bms.to(device), tgt.to(device)
            pred = model(drv, bms)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(tgt.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    # Inverse transform to original scale
    preds_orig = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    actuals_orig = tgt_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(actuals_orig, preds_orig)))
    mae = float(mean_absolute_error(actuals_orig, preds_orig))
    r2 = float(r2_score(actuals_orig, preds_orig))

    metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
    return metrics, preds_orig, actuals_orig


def extract_attention_weights(model, loader, device='cpu', max_batches=5):
    """
    Extract attention weights from CrossAttentionNet for a few batches.

    Returns:
        attn_bms_q: averaged (seq, seq) attention map (BMS queries Driving)
        attn_drv_q: averaged (seq, seq) attention map (Driving queries BMS)
    """
    model.eval()
    attn_bms_list = []
    attn_drv_list = []

    with torch.no_grad():
        for i, (drv, bms, tgt) in enumerate(loader):
            if i >= max_batches:
                break
            drv, bms = drv.to(device), bms.to(device)
            _, attn_bms, attn_drv = model(drv, bms, return_attn=True)
            # attn_bms: (batch, seq, seq) – mean across batch
            attn_bms_list.append(attn_bms.cpu().numpy().mean(axis=0))
            attn_drv_list.append(attn_drv.cpu().numpy().mean(axis=0))

    if not attn_bms_list:
        return None, None

    attn_bms_avg = np.mean(attn_bms_list, axis=0)
    attn_drv_avg = np.mean(attn_drv_list, axis=0)
    return attn_bms_avg, attn_drv_avg
