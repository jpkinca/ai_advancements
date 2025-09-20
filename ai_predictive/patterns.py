from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

def windowed_patterns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int = 20,
    stride: int = 1,
    normalize: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    arr = df[feature_cols].to_numpy(dtype=np.float32)
    n = len(arr)
    if n < window:
        return np.zeros((0, window * len(feature_cols)), dtype=np.float32), []
    feats = []
    meta: List[Dict] = []
    for start in range(0, n - window + 1, stride):
        end = start + window
        w = arr[start:end]
        if normalize:
            m = w.mean(axis=0, keepdims=True)
            s = w.std(axis=0, keepdims=True) + 1e-8
            w = (w - m) / s
        feats.append(w.reshape(-1))
        ts_start = df.index[start] if isinstance(df.index, pd.DatetimeIndex) else start
        ts_end = df.index[end - 1] if isinstance(df.index, pd.DatetimeIndex) else end - 1
        meta.append({"start": int(start), "end": int(end - 1), "ts_start": str(ts_start), "ts_end": str(ts_end)})
    X = np.asarray(feats, dtype=np.float32)
    return X, meta
