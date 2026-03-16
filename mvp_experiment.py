"""
MVP Experiment: EV Battery Energy Consumption Prediction
Cross-Attention based Multimodal Fusion vs. Baselines

Runs three models sequentially, saves results, figures, and JSON report.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt
import torch

# Make sure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import get_data_loaders
from models import build_model
from train import train_model
from evaluate import evaluate_model, extract_attention_weights

# ---- Config ----
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'bmw_i3_driving_cycles')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
JSON_OUT = os.path.join(RESULTS_DIR, 'final_results.json')

WINDOW_SIZE = 60
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_NAMES = ['LSTMBaseline', 'CNNLSTMConcat', 'CrossAttentionNet']

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    try:
        train_loader, val_loader, test_loader, scalers, stats = get_data_loaders(
            DATA_DIR,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE,
        )
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        print("Retrying with window_size=30...")
        train_loader, val_loader, test_loader, scalers, stats = get_data_loaders(
            DATA_DIR,
            window_size=30,
            stride=10,
            batch_size=BATCH_SIZE,
        )
        WINDOW_SIZE = 30

    print(f"\nData stats: {json.dumps(stats, indent=2)}\n")
    tgt_scaler = scalers['target']

    # ----------------------------------------------------------------
    # 2. Train and evaluate each model
    # ----------------------------------------------------------------
    all_histories = {}
    all_metrics = {}
    all_preds = {}
    all_actuals = {}

    for model_name in MODEL_NAMES:
        print("=" * 60)
        print(f"  Model: {model_name}")
        print("=" * 60)

        try:
            model = build_model(model_name, drv_dim=6, bms_dim=4, device=DEVICE)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {n_params:,}")

            history, best_val = train_model(
                model, train_loader, val_loader,
                epochs=EPOCHS, lr=LR, patience=PATIENCE,
                checkpoint_dir=CHECKPOINT_DIR,
                model_name=model_name, device=DEVICE,
            )

            metrics, preds, actuals = evaluate_model(
                model, test_loader, tgt_scaler, device=DEVICE
            )
            print(f"  Test  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")

            all_histories[model_name] = history
            all_metrics[model_name] = metrics
            all_preds[model_name] = preds
            all_actuals[model_name] = actuals

        except Exception as e:
            print(f"  [ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            # Record failure
            all_metrics[model_name] = {'rmse': None, 'mae': None, 'r2': None, 'error': str(e)}
            all_histories[model_name] = {'train_loss': [], 'val_loss': []}

    # ----------------------------------------------------------------
    # 3. Training curves figure
    # ----------------------------------------------------------------
    print("\nSaving training curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for ax, model_name, color in zip(axes, MODEL_NAMES, colors):
        hist = all_histories.get(model_name, {})
        train_l = hist.get('train_loss', [])
        val_l = hist.get('val_loss', [])
        if train_l:
            ax.plot(train_l, label='Train', color=color, linewidth=2)
            ax.plot(val_l, label='Val', color=color, linewidth=2, linestyle='--')
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves – EV Energy Prediction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    curve_path = os.path.join(FIGURES_DIR, 'training_curves.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {curve_path}")

    # ----------------------------------------------------------------
    # 4. Attention heatmap (CrossAttentionNet)
    # ----------------------------------------------------------------
    print("Extracting attention weights...")
    attn_bms_avg = None
    attn_drv_avg = None

    if 'CrossAttentionNet' in MODEL_NAMES and all_metrics.get('CrossAttentionNet', {}).get('rmse') is not None:
        try:
            ca_model = build_model('CrossAttentionNet', drv_dim=6, bms_dim=4, device=DEVICE)
            ckpt = os.path.join(CHECKPOINT_DIR, 'CrossAttentionNet_best.pt')
            if os.path.exists(ckpt):
                ca_model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            attn_bms_avg, attn_drv_avg = extract_attention_weights(
                ca_model, test_loader, device=DEVICE, max_batches=5
            )
        except Exception as e:
            print(f"  [WARN] Could not extract attention weights: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    seq = WINDOW_SIZE

    if attn_bms_avg is not None:
        im0 = axes[0].imshow(attn_bms_avg[:20, :20], aspect='auto', cmap='viridis')
        axes[0].set_title('Cross-Attn: BMS queries Driving\n(Q=BMS, K=V=Driving)', fontsize=10)
        axes[0].set_xlabel('Driving Time Step (Key)')
        axes[0].set_ylabel('BMS Time Step (Query)')
        plt.colorbar(im0, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, 'Attention not available', ha='center', va='center')
        axes[0].set_title('Cross-Attn: BMS queries Driving')

    if attn_drv_avg is not None:
        im1 = axes[1].imshow(attn_drv_avg[:20, :20], aspect='auto', cmap='magma')
        axes[1].set_title('Cross-Attn: Driving queries BMS\n(Q=Driving, K=V=BMS)', fontsize=10)
        axes[1].set_xlabel('BMS Time Step (Key)')
        axes[1].set_ylabel('Driving Time Step (Query)')
        plt.colorbar(im1, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, 'Attention not available', ha='center', va='center')
        axes[1].set_title('Cross-Attn: Driving queries BMS')

    plt.suptitle('Cross-Attention Weight Heatmaps', fontsize=13, fontweight='bold')
    plt.tight_layout()
    attn_path = os.path.join(FIGURES_DIR, 'attention_heatmap.png')
    plt.savefig(attn_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {attn_path}")

    # ----------------------------------------------------------------
    # 5. Save JSON results
    # ----------------------------------------------------------------
    output = {
        'experiment': 'MVP - EV Battery Energy Consumption Prediction',
        'data_stats': stats,
        'config': {
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'lr': LR,
            'device': DEVICE,
        },
        'models': {},
    }

    for model_name in MODEL_NAMES:
        m = all_metrics.get(model_name, {})
        h = all_histories.get(model_name, {})
        output['models'][model_name] = {
            'metrics': m,
            'n_epochs_trained': len(h.get('train_loss', [])),
            'best_val_loss': min(h.get('val_loss', [float('inf')])) if h.get('val_loss') else None,
        }

    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults JSON saved: {JSON_OUT}")

    # ----------------------------------------------------------------
    # 6. Summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print("-" * 60)
    for name in MODEL_NAMES:
        m = all_metrics.get(name, {})
        rmse = m.get('rmse')
        mae  = m.get('mae')
        r2   = m.get('r2')
        if rmse is not None:
            print(f"{name:<22} {rmse:>10.4f} {mae:>10.4f} {r2:>8.4f}")
        else:
            print(f"{name:<22} {'ERROR':>10} {'ERROR':>10} {'ERROR':>8}")
    print("=" * 60)

    return output


if __name__ == '__main__':
    result = main()
    print("\nExperiment complete.")
