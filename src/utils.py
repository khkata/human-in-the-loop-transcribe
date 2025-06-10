import os, tarfile, numpy as np, pandas as pd
import torchaudio
import re, unicodedata
from tqdm import tqdm
import jiwer
from jiwer.transforms import AbstractTransform
from fugashi import Tagger

# ── WER 前処理 ──
class NormalizeJapanese(AbstractTransform):
    def __call__(self, s):
        if isinstance(s, list):
            return [self._normalize(x) for x in s]
        return self._normalize(s)
    def _normalize(self, t):
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"[、。！!？?,.…‥]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

class TokenizeJapanese(AbstractTransform):
    def __init__(self):
        self.tagger = Tagger()
    def __call__(self, s):
        if isinstance(s, list):
            return [self._tokenize(x) for x in s]
        return self._tokenize(s)
    def _tokenize(self, text: str):
        return [tok.surface for tok in self.tagger(text)]

_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    NormalizeJapanese(),
    TokenizeJapanese()
])

def wer_ja(ref: str, hyp: str) -> float:
    if not ref or not ref.strip(): return np.nan
    w = jiwer.wer(ref, hyp,
                  truth_transform=_transform,
                  hypothesis_transform=_transform)
    return float(np.clip(w, 0.0, 1.0))

def ensure_clips(raw_audio, local_audio):
    if not os.path.isdir(local_audio) or not os.listdir(local_audio):
        os.makedirs(local_audio, exist_ok=True)
        with tarfile.open(raw_audio, "r:gz") as tar:
            tar.extractall(path=local_audio)

def preprocessing(raw_csv, processed_csv, raw_audio, local_audio, whisper_model):
    ensure_clips(raw_audio, local_audio)
    df = pd.read_csv(raw_csv)
    recs = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Transcribing"):
        path = f"{local_audio}/clips/{row.path}"
        res  = whisper_model.transcribe(path, language="ja", verbose=False)
        hyp  = (res.get("text") or "").strip()
        segs = res.get("segments") or []
        avg_lp = float(np.mean([s["avg_logprob"] for s in segs])) if segs else 0.0
        w = wer_ja(row.sentence, hyp)
        if np.isnan(w): continue
        recs.append({
            "client_id": row.client_id,
            "path": row.path,
            "sentence": row.sentence,
            "hyp_sentence": hyp,
            "wer": float(w),
            "avg_logprob": avg_lp,
        })
    out = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
    out.to_csv(processed_csv, index=False)
    return out
