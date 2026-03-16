"""
Data loader for BMW i3 driving cycle dataset (TripB series).
Handles CSV loading, feature extraction, normalization, windowing,
and PyTorch Dataset/DataLoader creation.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# Column names (semicolon-separated CSV)
DRIVING_FEATURES = [
    'Velocity [km/h]',
    'Throttle [%]',
    'Motor Torque [Nm]',
    'Longitudinal Acceleration [m/s^2]',
    'Regenerative Braking Signal ',   # note trailing space in header
    'Elevation [m]',
]

BATTERY_FEATURES = [
    'Battery Voltage [V]',
    'Battery Current [A]',
    'Battery Temperature [°C]',
    'SoC [%]',
]

TIME_COL = 'Time [s]'


def load_trip_csv(filepath):
    """Load a single trip CSV with semicolon separator, trying multiple encodings."""
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            df = pd.read_csv(filepath, sep=';', encoding=enc)
            # Strip whitespace from column names
            df.columns = [c.strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {filepath} with known encodings.")


def compute_window_energy(df, window_indices):
    """
    Compute energy Wh/km for a window of rows.
    energy_wh_per_km = sum(power_kw * dt * 1000) / sum(velocity_kmh * dt / 3600)
    Only non-zero velocity steps count toward distance.
    Returns NaN if total distance is zero.
    """
    rows = df.iloc[window_indices]
    time = rows['Time [s]'].values
    dt = np.diff(time, prepend=time[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0

    voltage = rows['Battery Voltage [V]'].values
    current = rows['Battery Current [A]'].values
    power_kw = voltage * np.abs(current) / 1000.0   # kW, always positive (discharge)

    velocity_kmh = rows['Velocity [km/h]'].values

    # Energy in Wh per step
    energy_wh = power_kw * dt / 3600.0 * 1000.0   # Wh

    # Distance in km per step
    dist_km = velocity_kmh * dt / 3600.0

    total_dist = dist_km.sum()
    if total_dist < 1e-6:
        return np.nan

    return energy_wh.sum() / total_dist


def process_trip(df, window_size=60, stride=10):
    """
    Given a loaded trip DataFrame, return:
      driving_windows: (N, window_size, 6)
      battery_windows: (N, window_size, 4)
      targets:         (N,)  [energy Wh/km per window]
    """
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Fix known column name quirk for Regenerative Braking Signal
    rbs_col = None
    for c in df.columns:
        if 'Regenerative Braking' in c:
            rbs_col = c
            break
    if rbs_col is None:
        raise ValueError("Regenerative Braking Signal column not found")

    drv_cols = [
        'Velocity [km/h]',
        'Throttle [%]',
        'Motor Torque [Nm]',
        'Longitudinal Acceleration [m/s^2]',
        rbs_col,
        'Elevation [m]',
    ]
    bms_cols = [
        'Battery Voltage [V]',
        'Battery Current [A]',
        'Battery Temperature [°C]',
        'SoC [%]',
    ]

    # Validate columns exist
    missing = [c for c in drv_cols + bms_cols + ['Time [s]'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop rows with NaN in required columns
    required = drv_cols + bms_cols + ['Time [s]']
    df = df[required].copy()
    df = df.dropna()
    df = df.reset_index(drop=True)

    n = len(df)
    if n < window_size:
        return None, None, None

    driving_windows = []
    battery_windows = []
    targets = []

    for start in range(0, n - window_size + 1, stride):
        indices = list(range(start, start + window_size))
        drv = df[drv_cols].iloc[indices].values.astype(np.float32)
        bms = df[bms_cols].iloc[indices].values.astype(np.float32)
        target = compute_window_energy(df, indices)

        if np.isnan(target) or target <= 0:
            continue

        driving_windows.append(drv)
        battery_windows.append(bms)
        targets.append(target)

    if len(targets) == 0:
        return None, None, None

    return (
        np.array(driving_windows, dtype=np.float32),
        np.array(battery_windows, dtype=np.float32),
        np.array(targets, dtype=np.float32),
    )


def load_all_trips(data_dir, window_size=60, stride=10):
    """
    Load all TripB CSV files, apply windowing, and return raw arrays.
    Returns list of (drv_windows, bms_windows, targets) per trip.
    """
    pattern = os.path.join(data_dir, 'TripB*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TripB CSV files found in {data_dir}")

    trip_data = []
    for f in files:
        try:
            df = load_trip_csv(f)
            drv, bms, tgt = process_trip(df, window_size=window_size, stride=stride)
            if drv is not None:
                trip_data.append((drv, bms, tgt))
            else:
                print(f"  Skipped (too short or no valid windows): {os.path.basename(f)}")
        except Exception as e:
            print(f"  Error loading {os.path.basename(f)}: {e}")

    return trip_data


def split_trips(trip_data, n_train=26, n_val=6, n_test=6):
    """Split trip list into train/val/test."""
    total = len(trip_data)
    # Adjust if we have fewer trips than expected
    n_test = min(n_test, total - n_train - n_val)
    if n_test < 0:
        n_val = max(1, total - n_train - 1)
        n_test = 1
        n_train = total - n_val - n_test

    train = trip_data[:n_train]
    val = trip_data[n_train:n_train + n_val]
    test = trip_data[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def fit_scalers(train_trips):
    """Fit MinMaxScaler on training data for driving and battery features."""
    drv_all = np.concatenate([t[0].reshape(-1, t[0].shape[-1]) for t in train_trips], axis=0)
    bms_all = np.concatenate([t[1].reshape(-1, t[1].shape[-1]) for t in train_trips], axis=0)
    tgt_all = np.concatenate([t[2] for t in train_trips], axis=0)

    drv_scaler = MinMaxScaler()
    bms_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()

    drv_scaler.fit(drv_all)
    bms_scaler.fit(bms_all)
    tgt_scaler.fit(tgt_all.reshape(-1, 1))

    return drv_scaler, bms_scaler, tgt_scaler


def apply_scalers(trips, drv_scaler, bms_scaler, tgt_scaler):
    """Apply fitted scalers to a list of trips."""
    scaled = []
    for drv, bms, tgt in trips:
        n, w, fd = drv.shape
        _, _, fb = bms.shape
        drv_s = drv_scaler.transform(drv.reshape(-1, fd)).reshape(n, w, fd).astype(np.float32)
        bms_s = bms_scaler.transform(bms.reshape(-1, fb)).reshape(n, w, fb).astype(np.float32)
        tgt_s = tgt_scaler.transform(tgt.reshape(-1, 1)).reshape(-1).astype(np.float32)
        scaled.append((drv_s, bms_s, tgt_s))
    return scaled


class EVDataset(Dataset):
    """PyTorch Dataset for EV energy prediction windows."""

    def __init__(self, trips):
        """
        trips: list of (drv_windows, bms_windows, targets)
               each already scaled
        """
        drv_list, bms_list, tgt_list = [], [], []
        for drv, bms, tgt in trips:
            drv_list.append(drv)
            bms_list.append(bms)
            tgt_list.append(tgt)

        self.drv = torch.tensor(np.concatenate(drv_list, axis=0), dtype=torch.float32)
        self.bms = torch.tensor(np.concatenate(bms_list, axis=0), dtype=torch.float32)
        self.tgt = torch.tensor(np.concatenate(tgt_list, axis=0), dtype=torch.float32)

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        return self.drv[idx], self.bms[idx], self.tgt[idx]


def get_data_loaders(data_dir, window_size=60, stride=10, batch_size=64,
                     n_train=26, n_val=6, n_test=6):
    """
    Full pipeline: load → split → scale → DataLoaders.
    Returns (train_loader, val_loader, test_loader, scalers, stats)
    """
    print(f"Loading TripB data from {data_dir}...")
    trip_data = load_all_trips(data_dir, window_size=window_size, stride=stride)
    print(f"  Loaded {len(trip_data)} valid trips")

    train_trips, val_trips, test_trips = split_trips(trip_data, n_train, n_val, n_test)
    print(f"  Train: {len(train_trips)} trips, Val: {len(val_trips)}, Test: {len(test_trips)}")

    drv_scaler, bms_scaler, tgt_scaler = fit_scalers(train_trips)

    train_scaled = apply_scalers(train_trips, drv_scaler, bms_scaler, tgt_scaler)
    val_scaled = apply_scalers(val_trips, drv_scaler, bms_scaler, tgt_scaler)
    test_scaled = apply_scalers(test_trips, drv_scaler, bms_scaler, tgt_scaler)

    train_ds = EVDataset(train_scaled)
    val_ds = EVDataset(val_scaled)
    test_ds = EVDataset(test_scaled)

    print(f"  Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Compute raw target stats (unscaled) for reporting
    raw_tgt = np.concatenate([t[2] for t in trip_data], axis=0)
    stats = {
        'n_trips': len(trip_data),
        'n_train_trips': len(train_trips),
        'n_val_trips': len(val_trips),
        'n_test_trips': len(test_trips),
        'total_windows': sum(len(t[2]) for t in trip_data),
        'target_mean': float(raw_tgt.mean()),
        'target_std': float(raw_tgt.std()),
        'target_min': float(raw_tgt.min()),
        'target_max': float(raw_tgt.max()),
        'window_size': window_size,
        'stride': stride,
    }

    scalers = {
        'driving': drv_scaler,
        'battery': bms_scaler,
        'target': tgt_scaler,
    }

    return train_loader, val_loader, test_loader, scalers, stats
