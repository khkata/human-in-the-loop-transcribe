#!/usr/bin/env python3
import argparse
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    AutoTokenizer, BertModel
)

from src.config import *
from src.utils  import preprocessing

class SpeechTextDataset(Dataset):
    def __init__(self, df, local_audio):
        self.df = df.reset_index(drop=True)
        self.local_audio = local_audio
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = torchaudio.load(f"{self.local_audio}/clips/{row.path}")
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        return wav.squeeze(0), row.hyp_sentence

def extract_and_cache(df, local_audio, features=None):
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained("rinna/japanese-wav2vec2-base")
    wav_enc  = Wav2Vec2Model.from_pretrained("rinna/japanese-wav2vec2-base").to(DEVICE).eval()
    bert_tok = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
    bert_enc = BertModel.from_pretrained("tohoku-nlp/bert-base-japanese-v3").to(DEVICE).eval()

    loader = DataLoader(
        SpeechTextDataset(df, local_audio),
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: ([w for w,_ in batch],[t for _,t in batch])
    )

    all_feats = []
    for wavs, texts in loader:
        wav_arrays = [w.cpu().numpy() for w in wavs]
        inputs = feat_ext(wav_arrays, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            wav_h = wav_enc(inputs.input_values.to(DEVICE)).last_hidden_state.mean(1).cpu()
        txt_inputs = bert_tok(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            txt_h = bert_enc(**{k:v.to(DEVICE) for k,v in txt_inputs.items()}).last_hidden_state.mean(1).cpu()
        all_feats.append(torch.cat([wav_h, txt_h], dim=-1))

    feats = torch.cat(all_feats, dim=0)
    if features is not None:
        torch.save({"_feats": feats}, features)
    return feats

def main():
    # 前処理（文字起こし＋WER 付与）
    df_train = preprocessing(
        TRAIN_RAW_CSV, TRAIN_PROCESSED_CSV,
        TRAIN_RAW_AUDIO, TRAIN_LOCAL_AUDIO,
        whisper_model
    )
    df_test = preprocessing(
        TEST_RAW_CSV,  TEST_PROCESSED_CSV,
        TEST_RAW_AUDIO,  TEST_LOCAL_AUDIO,
        whisper_model
    )

    # 特徴量抽出
    extract_and_cache(df_train, TRAIN_LOCAL_AUDIO, TRAIN_FEATURES)
    extract_and_cache(df_test,  TEST_LOCAL_AUDIO,  TEST_FEATURES)

if __name__ == "__main__":
    main()