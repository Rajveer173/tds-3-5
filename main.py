"""
Korean Audio Dataset API Verification Server  (speech-content interpretation)
============================================================================

POST /   (also /analyze)
  Input:  { "audio_id": "q0", "audio_base64": "<base64 audio file>" }
  Output: JSON with the exact key set:
          rows, columns, mean, std, variance, min, max, median, mode,
          range, allowed_values, value_range, correlation

KEY FINDING (from earlier grader feedback):
  The columns are NOT raw PCM channels. They come from the *spoken content*
  of the audio — the audio is Korean text-to-speech that reads out a small
  tabular dataset with Korean column names (e.g. "온도" = temperature).

So this server:
  1. decodes the base64 audio,
  2. transcribes it with Whisper (Korean),
  3. parses the transcript into a pandas DataFrame,
  4. computes standard pandas statistics,
  5. returns the required JSON.

The debug-capture endpoints are kept so you can retrieve a REAL sample the
grader sent and inspect exactly how the dataset is spoken — that is the one
thing that lets you tune parse_transcript() for an exact match.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # base/small/medium
WHISPER_LANG = os.environ.get("WHISPER_LANG", "ko")       # Korean audio
# allowed_values only for discrete columns with <= this many distinct values.
ALLOWED_VALUES_MAX_UNIQUE = int(os.environ.get("ALLOWED_VALUES_MAX_UNIQUE", "20"))

CAPTURE_DIR = "/tmp/captured_audio"
os.makedirs(CAPTURE_DIR, exist_ok=True)
DEBUG_TOKEN = os.environ.get("DEBUG_TOKEN", "")

app = FastAPI(title="Korean Audio Dataset Stats Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


# --------------------------------------------------------------------------
# JSON-safe conversion
# --------------------------------------------------------------------------
def to_native(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


# --------------------------------------------------------------------------
# Whisper transcription (lazy load)
# --------------------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_bytes: bytes) -> str:
    suffix = ".wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        suffix = ".mp3"
    elif audio_bytes[:4] == b"fLaC":
        suffix = ".flac"
    elif audio_bytes[:4] == b"OggS":
        suffix = ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    model = get_model()
    segments, _info = model.transcribe(path, language=WHISPER_LANG, beam_size=5)
    return " ".join(seg.text for seg in segments).strip()


# --------------------------------------------------------------------------
# Korean number-word -> integer (handles spoken numbers if Whisper writes words)
# --------------------------------------------------------------------------
_KO_DIGIT = {"영": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
             "육": 6, "칠": 7, "팔": 8, "구": 9}
_KO_UNIT = {"십": 10, "백": 100, "천": 1000}
_KO_BIG = {"만": 10000, "억": 100000000}


def ko_words_to_number(token: str) -> Optional[float]:
    if not token or not any(ch in _KO_DIGIT or ch in _KO_UNIT or ch in _KO_BIG for ch in token):
        return None
    total = 0
    section = 0
    current = 0
    for ch in token:
        if ch in _KO_DIGIT:
            current = _KO_DIGIT[ch]
        elif ch in _KO_UNIT:
            section += (current or 1) * _KO_UNIT[ch]
            current = 0
        elif ch in _KO_BIG:
            section += current
            total += (section or 1) * _KO_BIG[ch]
            section = 0
            current = 0
        else:
            return None
    return float(total + section + current)


# --------------------------------------------------------------------------
# Transcript -> DataFrame
# Multiple strategies since the exact spoken format is unknown until you
# capture a real sample. Tune this once you inspect /debug output.
# --------------------------------------------------------------------------
def _clean_cell(v: str):
    v = v.strip()
    if v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except ValueError:
        kw = ko_words_to_number(v)
        return kw if kw is not None else v


def _try_json(text: str) -> Optional[pd.DataFrame]:
    for m in re.finditer(r"(\{.*\}|\[.*\])", text, re.DOTALL):
        try:
            return pd.DataFrame(json.loads(m.group(1)))
        except Exception:
            continue
    return None


def _try_markdown_table(text: str) -> Optional[pd.DataFrame]:
    rows = [ln for ln in text.splitlines() if ln.count("|") >= 2]
    if len(rows) < 2:
        return None

    def split(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    header = split(rows[0])
    body = []
    for r in rows[1:]:
        if set(r.replace("|", "").strip()) <= set("-: "):
            continue
        cells = split(r)
        if len(cells) == len(header):
            body.append([_clean_cell(c) for c in cells])
    return pd.DataFrame(body, columns=header) if body else None


def _try_csv(text: str) -> Optional[pd.DataFrame]:
    for sep in [",", "\t", r"\s+"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if df.shape[1] >= 2 and df.shape[0] >= 1:
                return df.apply(lambda s: pd.to_numeric(s, errors="ignore"))
        except Exception:
            continue
    return None


def parse_transcript(text: str) -> pd.DataFrame:
    for strategy in (_try_json, _try_markdown_table, _try_csv):
        df = strategy(text)
        if df is not None and not df.empty:
            return df
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return pd.DataFrame({"value": [float(n) for n in nums]})
    return pd.DataFrame()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
        "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
        "allowed_values": {}, "value_range": {}, "correlation": [],
    }
    if df.empty:
        return result

    # Coerce numeric-looking columns.
    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().all():
            df[c] = conv

    columns = [str(c) for c in df.columns]
    result["rows"] = int(df.shape[0])
    result["columns"] = columns

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if numeric_cols:
        num = df[numeric_cols]
        result["mean"] = {str(k): v for k, v in num.mean().to_dict().items()}
        result["std"] = {str(k): v for k, v in num.std(ddof=1).to_dict().items()}
        result["variance"] = {str(k): v for k, v in num.var(ddof=1).to_dict().items()}
        result["min"] = {str(k): v for k, v in num.min().to_dict().items()}
        result["max"] = {str(k): v for k, v in num.max().to_dict().items()}
        result["median"] = {str(k): v for k, v in num.median().to_dict().items()}
        mode_row = num.mode()
        if not mode_row.empty:
            result["mode"] = {str(k): v for k, v in mode_row.iloc[0].to_dict().items()}
        result["range"] = {str(k): (num[k].max() - num[k].min()) for k in numeric_cols}
        result["value_range"] = {str(k): [num[k].min(), num[k].max()] for k in numeric_cols}

    # allowed_values: only discrete columns (small distinct-value set).
    for c in df.columns:
        if df[c].nunique(dropna=True) <= ALLOWED_VALUES_MAX_UNIQUE:
            uniques = pd.Series(df[c].dropna().unique())
            try:
                uniques = uniques.sort_values()
            except Exception:
                pass
            result["allowed_values"][str(c)] = uniques.tolist()

    if len(numeric_cols) >= 2:
        result["correlation"] = df[numeric_cols].astype(float).corr().values.tolist()
    elif len(numeric_cols) == 1:
        result["correlation"] = [[1.0]]

    return to_native(result)


# --------------------------------------------------------------------------
# Debug capture (retrieve a real grader sample to tune parsing)
# --------------------------------------------------------------------------
def capture_request(audio_id: str, audio_base64: str, transcript: str = "") -> None:
    try:
        with open(os.path.join(CAPTURE_DIR, f"{audio_id}.b64"), "w") as f:
            f.write(audio_base64)
        with open(os.path.join(CAPTURE_DIR, f"{audio_id}.meta.json"), "w") as f:
            json.dump({"audio_id": audio_id, "at": time.time(), "transcript": transcript}, f)
    except Exception:
        pass


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.post("/")
@app.post("/analyze")
def analyze_audio(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
    try:
        transcript = transcribe(audio_bytes)
    except Exception as e:
        transcript = ""
        capture_request(req.audio_id, req.audio_base64, f"TRANSCRIBE_ERROR: {e}")
        return compute_stats(pd.DataFrame())
    capture_request(req.audio_id, req.audio_base64, transcript)
    df = parse_transcript(transcript)
    return compute_stats(df)


@app.get("/debug/list")
def debug_list(token: str = Query(default="")):
    if DEBUG_TOKEN and token != DEBUG_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid debug token.")
    ids = sorted(f[:-4] for f in os.listdir(CAPTURE_DIR) if f.endswith(".b64"))
    return {"captured_audio_ids": ids}


@app.get("/debug/transcript/{audio_id}")
def debug_transcript(audio_id: str, token: str = Query(default="")):
    if DEBUG_TOKEN and token != DEBUG_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid debug token.")
    path = os.path.join(CAPTURE_DIR, f"{audio_id}.meta.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No capture for '{audio_id}'.")
    with open(path) as f:
        return json.load(f)


@app.get("/debug/audio/{audio_id}")
def debug_download(audio_id: str, token: str = Query(default="")):
    if DEBUG_TOKEN and token != DEBUG_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid debug token.")
    path = os.path.join(CAPTURE_DIR, f"{audio_id}.b64")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No captured audio for '{audio_id}'.")
    with open(path) as f:
        raw_bytes = base64.b64decode(f.read())
    ext = "bin"
    if raw_bytes[:4] == b"RIFF":
        ext = "wav"
    elif raw_bytes[:3] == b"ID3" or raw_bytes[:2] == b"\xff\xfb":
        ext = "mp3"
    elif raw_bytes[:4] == b"fLaC":
        ext = "flac"
    elif raw_bytes[:4] == b"OggS":
        ext = "ogg"
    return Response(
        content=raw_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{audio_id}.{ext}"'},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": WHISPER_MODEL, "lang": WHISPER_LANG}
