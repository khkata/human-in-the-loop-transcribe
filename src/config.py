# ==== インポート ====
import os
import torch
import whisper

# ==== 定数 ====
TRAIN_RAW_CSV        = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/raw/train/train.csv"
TRAIN_PROCESSED_CSV  = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/processed/train_processed.csv"
TRAIN_RAW_AUDIO      = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/raw/train/clips.tar.gz"
TRAIN_LOCAL_AUDIO    = "/tmp/train_clips"

TEST_RAW_CSV         = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/raw/test/test.csv"
TEST_PROCESSED_CSV   = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/processed/test_processed.csv"
TEST_RAW_AUDIO       = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/raw/test/clips.tar.gz"
TEST_LOCAL_AUDIO     = "/tmp/test_clips"

TRAIN_FEATURES       = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/processed/train_feats.pt"
TEST_FEATURES        = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/processed/test_feats.pt"

MODEL_DIR            = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/models/"
MODEL_PATH           = os.path.join(MODEL_DIR, "mlp_model.pt")
SCALER_PATH          = os.path.join(MODEL_DIR, "mlp_scaler.pt")
OPTUNA_PARAMS_PATH   = os.path.join(MODEL_DIR, "mlp_best_params.json")

TEST_HUMAN_CSV       = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/raw/test/test_human.csv"
TEST_INTEGRATED_CSV  = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/data/processed/test_integrated.csv"
OUTPUT_DIR           = "/content/drive/MyDrive/Human_in_the_Loop_Transcription/outputs"

DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR            = 16000
BATCH_SIZE           = 16

os.makedirs(os.path.dirname(TRAIN_PROCESSED_CSV), exist_ok=True)
os.makedirs(os.path.dirname(TRAIN_FEATURES),    exist_ok=True)
os.makedirs(os.path.dirname(TEST_PROCESSED_CSV), exist_ok=True)
os.makedirs(os.path.dirname(TEST_FEATURES),     exist_ok=True)
os.makedirs(MODEL_DIR,                          exist_ok=True)
os.makedirs(OUTPUT_DIR,                          exist_ok=True)

# ==== ASR モデル ====
whisper_model = whisper.load_model("small").to(DEVICE)