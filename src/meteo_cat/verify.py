"""Métricas de verificación y comprobaciones de los datos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def scores(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(obs) & np.isfinite(pred)
    if not ok.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    e = pred[ok] - obs[ok]
    return {"n": int(ok.sum()), "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e**2))), "bias": float(np.mean(e))}


def compare(df: pd.DataFrame, obs_col: str, pred_cols: dict[str, str], by: str | None = None) -> pd.DataFrame:
    """Tabla de métricas para varias predicciones, opcionalmente por grupos (p. ej. lead_day)."""
    rows = []
    groups = [(None, df)] if by is None else list(df.groupby(by))
    for g, sub in groups:
        for name, col in pred_cols.items():
            s = scores(sub[obs_col].to_numpy(dtype="float64"), sub[col].to_numpy(dtype="float64"))
            rows.append({"group": g, "prediction": name, **s})
    out = pd.DataFrame(rows)
    return out if by is not None else out.drop(columns="group")


def time_alignment(df: pd.DataFrame, obs_col: str, pred_col: str, max_shift_h: int = 3) -> dict[int, float]:
    """MAE de la predicción frente a la observación desplazada k horas, por estación.

    Si el mínimo no está en 0, las horas de las observaciones no están en UTC (o hay otro desfase).
    """
    out = {}
    base = df[["station", "time", obs_col]].copy()
    for k in range(-max_shift_h, max_shift_h + 1):
        shifted = base.assign(time=base["time"] + pd.Timedelta(hours=k))
        m = df[["station", "time", pred_col]].merge(shifted, on=["station", "time"])
        out[k] = scores(m[obs_col].to_numpy(dtype="float64"), m[pred_col].to_numpy(dtype="float64"))["mae"]
    return out


def prob_scores(obs: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Brier, Brier de la climatología, habilidad (BSS) y AUC de una probabilidad frente a 0/1."""
    ok = np.isfinite(obs) & np.isfinite(prob)
    o, p = obs[ok].astype(float), np.clip(prob[ok], 0, 1)
    if len(o) == 0 or o.min() == o.max():
        return {"n": int(len(o)), "brier": float("nan"), "bss": float("nan"), "auc": float("nan"),
                "frecuencia": float(o.mean()) if len(o) else float("nan")}
    brier = float(np.mean((p - o) ** 2))
    clim = float(np.mean((o.mean() - o) ** 2))
    # AUC con la fórmula de Mann-Whitney (rangos medios para empates).
    ranks = pd.Series(p).rank().to_numpy()
    n1 = o.sum()
    auc = float((ranks[o == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(o) - n1)))
    return {"n": int(len(o)), "brier": brier, "bss": 1 - brier / clim, "auc": auc, "frecuencia": float(o.mean())}


def reliability(obs: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict]:
    """Frecuencia observada por tramos de probabilidad prevista (bien calibrado: diagonal)."""
    ok = np.isfinite(obs) & np.isfinite(prob)
    o, p = obs[ok], np.clip(prob[ok], 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"tramo": f"{b * 100 // bins}-{(b + 1) * 100 // bins}%", "n": int(m.sum()),
                        "prevista": float(p[m].mean()), "observada": float(o[m].mean())})
    return out


def persistence_tau(df: pd.DataFrame, resid_col: str, max_lag_h: int = 12, lag_step_h: int = 1) -> float:
    """Escala temporal (h) con que se pierde la memoria del error en estaciones.

    Ajusta corr(lag) ≈ exp(-lag/tau). Se usa en el nowcasting para decidir cuánto
    pesan las últimas observaciones según el horizonte.
    """
    s = df[["station", "time", resid_col]].dropna().drop_duplicates(["station", "time"])
    s = s.set_index(["station", "time"])[resid_col]
    lags, corrs = [], []
    for lag in range(lag_step_h, max_lag_h + 1, lag_step_h):
        a = s.reset_index()
        b = a.assign(time=a["time"] + pd.Timedelta(hours=lag))
        m = a.merge(b, on=["station", "time"], suffixes=("", "_lag"))
        if len(m) < 100:
            continue
        c = np.corrcoef(m[resid_col], m[resid_col + "_lag"])[0, 1]
        if c > 0.05:
            lags.append(lag)
            corrs.append(c)
    if len(lags) < 2:
        return 3.0
    # Ajuste por mínimos cuadrados de log(corr) = -lag/tau (recta por el origen).
    lags_a, logc = np.array(lags, float), np.log(np.array(corrs))
    slope = np.sum(lags_a * logc) / np.sum(lags_a**2)
    return float(np.clip(-1.0 / slope, 0.5, 24.0)) if slope < 0 else 24.0
