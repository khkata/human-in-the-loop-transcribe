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
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
TMP_DIR         = "/tmp"

# ── splitごとのパスを一括定義 ──
splits = ["train", "test"]
paths = {}

for split in splits:
    paths[f"{split}_raw_csv"]        = os.path.join(RAW_DIR, split, f"{split}.csv")
    paths[f"{split}_processed_csv"]  = os.path.join(PROCESSED_DIR, f"{split}_processed.csv")
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
dirs_to_make = {os.path.dirname(p) for p in paths.values()}
dirs_to_make.add(OUTPUT_DIR)
for d in dirs_to_make:
    os.makedirs(d, exist_ok=True)

DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR            = 16000
BATCH_SIZE           = 16

# ==== ASR モデル ====
whisper_model = whisper.load_model("small").to(DEVICE)

# ==== top-level aliases for easier imports ====
TRAIN_RAW_CSV       = paths["train_raw_csv"]
TRAIN_PROCESSED_CSV = paths["train_processed_csv"]
TRAIN_RAW_AUDIO     = paths["train_raw_audio"]
TRAIN_LOCAL_AUDIO   = paths["train_local_audio"]
TRAIN_FEATURES      = paths["train_features"]

TEST_RAW_CSV        = paths["test_raw_csv"]
TEST_PROCESSED_CSV  = paths["test_processed_csv"]
TEST_RAW_AUDIO      = paths["test_raw_audio"]
TEST_LOCAL_AUDIO    = paths["test_local_audio"]
TEST_FEATURES       = paths["test_features"]

TEST_HUMAN_CSV      = paths["test_human_csv"]