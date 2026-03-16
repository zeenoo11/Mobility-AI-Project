"""
Hyperparameter Search for CrossAttentionNet
Grid search over key hyperparameters, saves results to JSON.
"""

import os
import sys
import json
import itertools
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import get_data_loaders
from models import build_model
from train import train_model
from evaluate import evaluate_model

# ---- Config ----
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'bmw_i3_driving_cycles')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'hparam_search')
JSON_OUT = os.path.join(RESULTS_DIR, 'hparam_search.json')

WINDOW_SIZE = 60
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WARMUP_EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Search Space ----
SEARCH_SPACE = {
    'lstm_hidden': [64, 128],
    'n_heads': [4, 8],
    'dropout': [0.1, 0.2, 0.3],
    'cnn_out': [48, 96],
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def run_single_config(config, train_loader, val_loader, test_loader, tgt_scaler):
    """Train and evaluate a single hyperparameter configuration."""
    config_str = '_'.join(f'{k}{v}' for k, v in sorted(config.items()))
    model_name = f'CA_{config_str}'

    print(f"\n{'='*60}")
    print(f"  Config: {config}")
    print(f"{'='*60}")

    model = build_model('CrossAttentionNet', drv_dim=6, bms_dim=4,
                         device=DEVICE, **config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    history, best_val = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        warmup_epochs=WARMUP_EPOCHS,
        scheduler_type='cosine_warmup',
        checkpoint_dir=CHECKPOINT_DIR,
        model_name=model_name, device=DEVICE,
        verbose=True,
    )
    train_time = time.time() - t0

    metrics, preds, actuals = evaluate_model(
        model, test_loader, tgt_scaler, device=DEVICE
    )

    result = {
        'config': config,
        'n_params': n_params,
        'n_epochs_trained': len(history['train_loss']),
        'best_val_loss': float(best_val),
        'train_time_sec': round(train_time, 1),
        'test_rmse': metrics['rmse'],
        'test_mae': metrics['mae'],
        'test_r2': metrics['r2'],
    }

    print(f"  Result: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
          f"R²={metrics['r2']:.4f}  ({train_time:.0f}s)")

    return result


def main():
    print(f"Device: {DEVICE}")
    print(f"Search space: {SEARCH_SPACE}")

    # Generate all combinations
    keys = sorted(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    print(f"Total configurations: {len(configs)}")

    # Load data once
    train_loader, val_loader, test_loader, scalers, stats = get_data_loaders(
        DATA_DIR, window_size=WINDOW_SIZE, stride=STRIDE, batch_size=BATCH_SIZE,
    )
    tgt_scaler = scalers['target']

    # Run grid search
    all_results = []
    for i, config in enumerate(configs):
        print(f"\n>>> Configuration {i+1}/{len(configs)}")
        try:
            result = run_single_config(
                config, train_loader, val_loader, test_loader, tgt_scaler
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results.append({
                'config': config,
                'error': str(e),
            })

        # Save intermediate results
        output = {
            'experiment': 'Hyperparameter Search - CrossAttentionNet',
            'search_space': SEARCH_SPACE,
            'training_config': {
                'epochs': EPOCHS, 'patience': PATIENCE, 'lr': LR,
                'warmup_epochs': WARMUP_EPOCHS, 'scheduler': 'cosine_warmup',
            },
            'data_stats': stats,
            'results': all_results,
        }
        with open(JSON_OUT, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)

    valid = [r for r in all_results if 'test_r2' in r]
    if not valid:
        print("No valid results.")
        return

    valid_sorted = sorted(valid, key=lambda r: r['test_r2'], reverse=True)

    print(f"\n{'Rank':<5} {'lstm_h':<8} {'heads':<6} {'drop':<6} {'cnn':<6} "
          f"{'Params':<10} {'Epochs':<7} {'RMSE':<10} {'MAE':<10} {'R²':<8}")
    print("-" * 80)

    for rank, r in enumerate(valid_sorted, 1):
        c = r['config']
        print(f"{rank:<5} {c['lstm_hidden']:<8} {c['n_heads']:<6} {c['dropout']:<6.1f} "
              f"{c['cnn_out']:<6} {r['n_params']:<10,} {r['n_epochs_trained']:<7} "
              f"{r['test_rmse']:<10.4f} {r['test_mae']:<10.4f} {r['test_r2']:<8.4f}")

    best = valid_sorted[0]
    print(f"\nBest config: {best['config']}")
    print(f"Best R²: {best['test_r2']:.4f}, RMSE: {best['test_rmse']:.4f}, "
          f"MAE: {best['test_mae']:.4f}")

    # Save final results
    output['best_config'] = best['config']
    output['best_metrics'] = {
        'rmse': best['test_rmse'],
        'mae': best['test_mae'],
        'r2': best['test_r2'],
    }
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {JSON_OUT}")


if __name__ == '__main__':
    main()
