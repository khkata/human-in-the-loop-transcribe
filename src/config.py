# ==== インポート ====
import os
import torch
import whisper

# ==== 定数 ====
# ── ベースディレクトリ定義 ──
BASE_DIR        = "/content/human-in-the-loop-transcribe"
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
TMP_DIR         = "/tmp"

# ── splitごとのパスを一括定義 ──
splits = ["train", "test"]
paths = {}

for split in splits:
    paths[f"{split}_raw_csv"]        = os.path.join(RAW_DIR, split, f"{split}.csv")
    paths[f"{split}_processed_csv"]  = os.path.join(PROCESSED_DIR, split, f"{split}_processed.csv")
    paths[f"{split}_raw_audio"]      = os.path.join(RAW_DIR, split, "clips.tar.gz")
    paths[f"{split}_local_audio"]    = os.path.join(TMP_DIR, f"{split}_clips")
    paths[f"{split}_features"]       = os.path.join(PROCESSED_DIR, f"{split}_feats.pt")

# test_human.csv は train/test ループ外で追加
paths["test_human_csv"] = os.path.join(RAW_DIR, "test", "test_human.csv")

# ── モデル関連パス ──
paths["model_path"]         = os.path.join(MODEL_DIR, "mlp_model.pt")
paths["scaler_path"]        = os.path.join(MODEL_DIR, "mlp_scaler.pt")
paths["optuna_params_path"] = os.path.join(MODEL_DIR, "mlp_best_params.json")

# ── ディレクトリの一括作成 ──
# 各パスの親ディレクトリを集めて makedirs
dirs_to_make = {os.path.dirname(p) for p in paths.values()}
for d in dirs_to_make:
    os.makedirs(d, exist_ok=True)

os.makedirs(os.path.dirname("outputs"), exist_ok=True)

# ==== ASR モデル ====
whisper_model = whisper.load_model("small").to(DEVICE)