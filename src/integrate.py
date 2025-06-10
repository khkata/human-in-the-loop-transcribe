#!/usr/bin/env python3
import os
import re
import unicodedata
import argparse
import numpy as np
import pandas as pd
import fugashi
import jiwer
from jiwer.transforms import AbstractTransform
from src.config import (
    TEST_HUMAN_CSV, TEST_PROCESSED_CSV, TEST_INTEGRATED_CSV
)

# [不明]除去用パターン
UNKNOWN_PATTERN = (
    r"[\[\]「」『』\\\//\(\)\{\}＜＞〈〉【】]*\s*不明\s*"
    r"[\[\]「」『』\\\//\(\)\{\}＜＞〈〉【】]*"
)

# ==================== 日本語 WER 計算用変換 ====================
class NormalizeJapanese(AbstractTransform):
    def __call__(self, s):
        if isinstance(s, list):
            return [self._normalize(x) for x in s]
        return self._normalize(s)

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[、。！!？?,.…‥]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

class TokenizeJapanese(AbstractTransform):
    def __init__(self):
        self.tagger = fugashi.Tagger()
    def __call__(self, s):
        if isinstance(s, list):
            return [self._tokenize(x) for x in s]
        return self._tokenize(s)
    def _tokenize(self, text: str) -> list:
        return [tok.surface for tok in self.tagger(text)]

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    NormalizeJapanese(),
    TokenizeJapanese()
])

def wer_ja(ref: str, hyp: str) -> float:
    if not isinstance(ref, str) or not ref.strip():
        return np.nan
    w = jiwer.wer(ref, hyp,
                   truth_transform=transform,
                   hypothesis_transform=transform)
    return float(np.clip(w, 0.0, 1.0))

# ==================== ROVER-3 多数決統合 ====================
def fuse_transcriptions(row: pd.Series) -> str:
    texts = [row.get(c, "") or "" for c in ("human_a", "human_b", "human_c")]
    tokens_list = [t.split() for t in texts]
    max_len = max(len(t) for t in tokens_list)
    fused = []
    for i in range(max_len):
        votes = {}
        for t in tokens_list:
            if i < len(t) and t[i]:
                votes[t[i]] = votes.get(t[i], 0) + 1
        if votes:
            fused.append(max(votes, key=votes.get))
    return " ".join(fused)

# ==================== 正規化処理 ====================
def normalize_unknowns(df: pd.DataFrame, columns: list, pattern: str) -> pd.DataFrame:
    for col in columns:
        df[col] = (
            df[col].astype(str)
                   .apply(lambda x: re.sub(pattern, "", x).strip())
        )
    return df

# ==================== 統合処理 ====================
def main(args):
    # 人手データ読み込み・統合
    df_h = pd.read_csv(args.human_csv)
    df_h = normalize_unknowns(df_h, ["human_a", "human_b", "human_c"], UNKNOWN_PATTERN)
    df_h["human_sentence"] = df_h.apply(fuse_transcriptions, axis=1)
    df_h["human_wer"] = df_h.apply(lambda x: wer_ja(x["sentence"], x["human_sentence"]), axis=1)

    # ASR結果読み込み
    df_p = pd.read_csv(args.processed_csv)

    # 結合 DataFrame 作成
    df_out = pd.DataFrame({
        "client_id":      df_p["client_id"],
        "path":           df_p["path"],
        "sentence":       df_p["sentence"],
        "asr_sentence":   df_p["hyp_sentence"],
        "asr_wer":        df_p["wer"],
        "human_sentence": df_h["human_sentence"],
        "human_wer":      df_h["human_wer"],
        "avg_logprob":    df_p["avg_logprob"],
        "wer_pred":       df_p["wer_pred"]
    })

    # 出力
    os.makedirs(os.path.dirname(args.integrated_csv), exist_ok=True)
    df_out.to_csv(args.integrated_csv, index=False)
    print(f"Integrated file saved: {args.integrated_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROVER-3 majority-vote integration for ASR/human transcripts.")
    parser.add_argument("--human_csv",      default=TEST_HUMAN_CSV,      help="Path to CSV with human transcripts (columns human_a,b,c).")
    parser.add_argument("--processed_csv",  default=TEST_PROCESSED_CSV,  help="Path to ASR processed CSV.")
    parser.add_argument("--integrated_csv", default=TEST_INTEGRATED_CSV, help="Output path for integrated CSV.")
    args = parser.parse_args()
    main(args)