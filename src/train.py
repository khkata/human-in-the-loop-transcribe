#!/usr/bin/env python3
import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import optuna
import warnings

from src.config import (
    TRAIN_PROCESSED_CSV, TRAIN_FEATURES,
    MODEL_DIR, MODEL_PATH, SCALER_PATH, OPTUNA_PARAMS_PATH,
    DEVICE, BATCH_SIZE, MAX_TRIALS, PATIENCE_OS,
    PATIENCE_FULL, MAX_EPOCHS_OS, MAX_EPOCHS_FULL, N_SPLITS
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna\\.distributions"
)

# ── 乱数シード ──
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

# ── モデル定義 ──
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_prob=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_prob)
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── データセット ──
class WERDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: np.ndarray, weights: np.ndarray = None):
        self.X = features
        self.y = torch.from_numpy(targets.astype(np.float32))
        self.weights = None
        if weights is not None:
            self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.weights is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.weights[idx]

# ── サンプラー作成 ──
def create_weighted_sampler(targets: np.ndarray, num_bins: int = 10):
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idx = np.digitize(targets, bins) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=num_bins)
    weight_bin = np.where(bin_counts > 0, 1.0 / bin_counts, 0.0)
    sample_weights = weight_bin[bin_idx]
    sample_weights /= sample_weights.sum()
    return sample_weights

# ── データロード & 前処理 ──
def load_data_and_features():
    # CSV 読み込み
    df = pd.read_csv(TRAIN_PROCESSED_CSV, encoding="utf-8")
    df["wer"] = df["wer"].fillna(1.0).clip(0.0, 1.0)

    # 特徴量ロード
    cache = torch.load(TRAIN_FEATURES, map_location="cpu")
    feats = cache["_feats"]

    # 標準化
    scaler = StandardScaler()
    feats_np = feats.numpy()
    feats_norm = scaler.fit_transform(feats_np)

    # スケーラー保存
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    torch.save({
        'mean': torch.from_numpy(scaler.mean_).unsqueeze(0),
        'std':  torch.from_numpy(scaler.scale_).unsqueeze(0)
    }, SCALER_PATH)

    feats_t = torch.from_numpy(feats_norm).float()
    targets = df["wer"].values.astype(np.float32)
    groups  = df.get("client_id", pd.Series()).values
    return feats_t, targets, groups

# ── Optuna 目的関数 ──
def objective(trial):
    feats, targets, groups = load_data_and_features()
    gkf = GroupKFold(n_splits=N_SPLITS)
    train_idx, val_idx = next(gkf.split(feats, targets, groups))

    # ハイパラ探索領域
    hidden  = trial.suggest_categorical("hidden_sizes", [
        (512,256,128), (1024,512,256),
        (512,512,256,128), (1024,512,256,128),
        (1024,512,256,128,64)
    ])
    lr      = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    wd      = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # サンプラー＋データローダ
    train_w = create_weighted_sampler(targets[train_idx])
    sampler = WeightedRandomSampler(train_w, len(train_w), replacement=True)
    train_ds = WERDataset(feats[train_idx], targets[train_idx], train_w)
    val_ds   = WERDataset(feats[val_idx],   targets[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # モデル・最適化
    model     = MLPRegressor(feats.shape[1], hidden, dropout).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_OS, eta_min=1e-6)

    best_val, patience = float('inf'), 0
    for epoch in range(1, MAX_EPOCHS_OS+1):
        # train
        model.train()
        for batch in train_loader:
            Xb, yb, *rest = batch
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # val
        model.eval()
        losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                losses.append(criterion(model(Xb), yb).item())
        val_loss = np.mean(losses)

        if val_loss < best_val:
            best_val  = val_loss
            patience  = 0
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        else:
            patience += 1

        trial.report(val_loss, epoch)
        if trial.should_prune() or patience >= PATIENCE_OS:
            break

    model.load_state_dict(best_state)
    return best_val

# ── メイン ──
def main(args):
    set_seed(args.seed)

    # Optuna 探索
    os.makedirs(MODEL_DIR, exist_ok=True)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)

    # ベストパラ保存
    with open(OPTUNA_PARAMS_PATH, 'w', encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)
    print("Best params saved to", OPTUNA_PARAMS_PATH)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--trials", type=int, default=MAX_TRIALS)
    args = p.parse_args()
    main(args)