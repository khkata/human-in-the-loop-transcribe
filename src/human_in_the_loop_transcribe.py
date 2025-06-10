import os
import argparse
import torch
import torchaudio
import numpy as np
import pandas as pd
import subprocess
import whisper
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoTokenizer, BertModel
import torch.nn as nn
from IPython.display import Audio, display, clear_output
import ipywidgets as widgets
from torch.utils.data import Dataset, DataLoader

from src.config import *
from src.extract_feats import SpeechTextDataset, extract_and_cache
from src.models import MLPRegressor

# 音声をWhisperで分割し、データフレームで返す
def split_audio_segments(audio_path: str, clips_dir: str, overlap: float = 1.0) -> pd.DataFrame:
    os.makedirs(clips_dir, exist_ok=True)

    print("Whisperで文字起こし中...")
    result = whisper_model.transcribe(audio_path, language="ja", fp16=False, verbose=False)
    segments = result["segments"]

    audio_duration = segments[-1]['end'] if segments else 0.0

    records = []
    for idx, seg in enumerate(segments, start=1):
        start = max(0.0, seg["start"] - overlap)
        end   = min(audio_duration, seg["end"] + overlap)
        text  = seg["text"].strip()
        seg_id = f"{idx:04d}"
        out_path = f"segment_{seg_id}.wav"
        full_path = f"{clips_dir}/{out_path}"

        # ffmpegで切り出し（-ss 開始位置, -to 終了位置）
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-to", str(end),
            "-ac", "1", "-ar", "16000", full_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        records.append({
            "full_path"         : full_path,
            "path"              : out_path,
            "start"             : start,
            "end"               : end,
            "hyp_sentence"      : text,
        })

    return pd.DataFrame(records)

def wer_estimation(df, feats, scaler, model):
    print("WER予測中...")

    # スケーリング
    scaler = torch.load(scaler)
    mean_vec = scaler['mean']
    std_vec = scaler['std']
    feats_norm = ((feats - mean_vec) / std_vec).float()

    # モデルロード
    ckpt = torch.load(model, map_location=DEVICE)
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

    # 結果をデータフレームに追加
    df['wer_pred'] = preds

    return df

# WERが閾値以上のセグメントをユーザーにレビューしてもらう
def collect_user_transcriptions(df, threshold, orig_df, output_dir):
    high_df = df[df['wer_pred'] >= threshold].copy()
    high_df['user_transcription'] = ""
    high_df['delete'] = False

    display(widgets.HTML(
        "<div style='margin-bottom:1em; color:#b33;'>"
        "<b>自動音声認識の誤認識が多いと推定される音声です。<br>"
        "文字起こしを入力してください。。<br>"
        "不要な音声は「削除」にチェックを入れてください。<br>"
        "全ての入力が終わったら「完了」ボタンをクリックしてください。</b>"
        "</div>"
    ))

    for idx, row in high_df.iterrows():
        display(widgets.HTML(f"<b>セグメント {row['path']} (WER={row['wer_pred']:.2f})</b>"))
        display(Audio(row['full_path'], rate=16000))
        display(widgets.HTML(f"<i>Whisperによる文字起こし:</i> {row['hyp_sentence']}"))

        text_input = widgets.Text(
            placeholder="ここに文字起こしを入力",
            description="テキスト：",
            layout=widgets.Layout(width="75%")
        )
        def on_text_change(change, ix=idx):
            high_df.at[ix, 'user_transcription'] = change['new']
        text_input.observe(on_text_change, names='value')
        display(text_input)

        delete_cb = widgets.Checkbox(
            value=False,
            description="この音声セグメントを削除",
            indent=False
        )
        def on_delete_change(change, ix=idx):
            high_df.at[ix, 'delete'] = change['new']
        delete_cb.observe(on_delete_change, names='value')
        display(delete_cb)

        display(widgets.HTML("<hr>"))

    save_btn = widgets.Button(description="完了・保存", button_style="success")
    out = widgets.Output()
    def on_save(btn):
        with out:
            clear_output()
            print("CSV と TXTを出力しました。")
        merge_and_save(orig_df, high_df[['path','user_transcription','delete']], output_dir)

    save_btn.on_click(on_save)
    display(save_btn, out)

# ASRとユーザー文字起こしをマージし、CSVとTXTを出力
def merge_and_save(orig_df, user_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = orig_df.merge(user_df, on='path', how='left')
    df = df[df['delete'] != True].copy()

    df['merged_transcription'] = df.apply(
        lambda row: row['user_transcription'].strip()
                    if pd.notnull(row['user_transcription']) and row['user_transcription'].strip()
                    else row['hyp_sentence'],
        axis=1
    )
    df_out  = df[['path', 'hyp_sentence', 'user_transcription', 'merged_transcription']]
    df_out  = df.rename({'hyp_sentence': 'asr_transcription'}, axis='columns')

    # CSV と TXT 出力
    csv_path = os.path.join(output_dir, 'merged_transcriptions.csv')
    txt_path = os.path.join(output_dir, 'merged_transcriptions.txt')
    df_out.to_csv(csv_path, index=False, encoding='utf-8')

    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in df_out['merged_transcription']:
            f.write(f"{line}\n")

def main(args):
    segments_df = split_audio_segments(args.audio, args.clips_dir)
    feats = extract_and_cache(segments_df, args.clips_dir)
    df = wer_estimation(segments_df, feats, args.scaler, args.model)
    clear_output()
    collect_user_transcriptions(df, args.threshold,
                                orig_df=df,
                                output_dir=args.output_dir)

# メイン処理
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Human-in-the-Loop-Transcribe")
    parser.add_argument("--audio",      default="/content/input_audio.wav")
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument("--clips_dir",  default="/tmp")
    parser.add_argument("--scaler", default=SCALER_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args)