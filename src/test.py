#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from src.config import (
    TEST_FEATURES, SCALER_PATH, MODEL_PATH, TEST_PROCESSED_CSV,
    DEVICE
)
from src.models import MLPRegressor

def evaluate(test_features: str, scaler_path: str, model_path: str, processed_csv: str):
    # 特徴量ロード
    cache = torch.load(test_features, map_location="cpu")
    feats = cache.get("_feats")

    # スケーリング
    scaler = torch.load(scaler_path)
    mean_vec = scaler['mean']
    std_vec = scaler['std']
    feats_norm = ((feats - mean_vec) / std_vec).float()

    # モデルロード
    ckpt = torch.load(model_path, map_location=DEVICE)
    mlp = MLPRegressor(
        input_dim=ckpt['input_dim'],
        hidden_sizes=tuple(ckpt['hidden_sizes']),
        dropout_prob=ckpt['dropout_prob']
    ).to(DEVICE)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()

    # 推論
    with torch.no_grad():
        preds = mlp(feats_norm.to(DEVICE)).cpu().numpy().clip(0.0, 1.0)

    # 結果を CSV に追記
    df = pd.read_csv(processed_csv, encoding="utf-8")
    df['wer_pred'] = preds
    df.to_csv(processed_csv, index=False)
    print(f"Saved predictions to {processed_csv}")

    # 評価指標計算
    y_true = df['wer'].values
    y_pred = df['wer_pred'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"PCC  = {pcc:.4f}")
    print(f"ρ    = {rho:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP WER predictions and compute metrics.")
    parser.add_argument("--test_features", default=TEST_FEATURES,
                        help="Path to .pt file containing test features.")
    parser.add_argument("--scaler_path", default=SCALER_PATH,
                        help="Path to scaler .pt file.")
    parser.add_argument("--model_path", default=MODEL_PATH,
                        help="Path to trained model .pt file.")
    parser.add_argument("--processed_csv", default=TEST_PROCESSED_CSV,
                        help="Path to processed test CSV with ground-truth WER.")
    args = parser.parse_args()

    evaluate(
        args.test_features,
        args.scaler_path,
        args.model_path,
        args.processed_csv
    )

if __name__ == '__main__':
    main()