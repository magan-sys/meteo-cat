"""Construcción de la tabla de predictores, común a entrenamiento y operación.

Entrada "ancha": una fila por (lugar, hora válida) con columnas `<var>__<modelo>` y `lead_h`.
El modelo ML aprende el residuo respecto a una base física:
    base = media de modelos + gradiente térmico estándar * (altitud real - altitud a escala de modelo)
    objetivo = observación - base
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sources.nwp import VARIABLES
from .terrain import TERRAIN_FEATURES

LAPSE_RATE = -0.0065  # °C/m, atmósfera estándar
TARGET_VAR = "t2m"


def _nanmean(a: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=1)


def build_features(wide: pd.DataFrame, terrain: dict[str, np.ndarray],
                   models: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray]:
    """Devuelve (X, base). `terrain` contiene arrays alineados con las filas de `wide`."""
    X = pd.DataFrame(index=wide.index)

    def col(v, m):
        c = f"{v}__{m}"
        return wide[c].to_numpy(dtype="float32") if c in wide else np.full(len(wide), np.nan, "float32")

    for m in models:
        X[f"t2m__{m}"] = col("t2m", m)
        X[f"d2m__{m}"] = col("d2m", m)
        X[f"tcc__{m}"] = col("tcc", m)
        X[f"ws10__{m}"] = np.hypot(col("u10", m), col("v10", m))

    t = X[[f"t2m__{m}" for m in models]].to_numpy(dtype="float64")
    ens_mean = _nanmean(t)
    with np.errstate(all="ignore"):
        X["ens_std"] = np.nanstd(t, axis=1) if len(models) > 1 else 0.0
    X["ens_mean"] = ens_mean
    X["n_models"] = np.isfinite(t).sum(axis=1)
    for v in ("d2m", "tcc"):
        X[f"{v}__mean"] = _nanmean(X[[f"{v}__{m}" for m in models]].to_numpy(dtype="float64"))
    X["spread_dew"] = X["ens_mean"] - X["d2m__mean"]
    for v in ("u10", "v10"):
        X[f"{v}__mean"] = _nanmean(np.column_stack([col(v, m) for m in models]))
    X["ws10__mean"] = np.hypot(X["u10__mean"], X["v10__mean"])

    time = pd.to_datetime(wide["time"], utc=True)
    hour = time.dt.hour.to_numpy() + time.dt.minute.to_numpy() / 60.0
    doy = time.dt.dayofyear.to_numpy()
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["lead_h"] = wide["lead_h"].to_numpy(dtype="float32")

    for k in TERRAIN_FEATURES:
        X[k] = np.asarray(terrain[k], dtype="float32")
    dz = X["elevation"].to_numpy() - X["elev_smooth_25km"].to_numpy()
    X["elev_minus_smooth"] = dz

    base = ens_mean + LAPSE_RATE * dz
    return X, base


RAIN_THRESHOLD_MM = 0.2  # lluvia apreciable en 3 h (la resolución de los pluviómetros es 0,1 mm)


def build_rain_features(wide: pd.DataFrame, terrain: dict[str, np.ndarray],
                        models: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray]:
    """Predictores de lluvia en 3 h. Devuelve (X, prob_bruta).

    `prob_bruta` es la fracción de modelos que dan lluvia apreciable: la probabilidad que se
    obtendría sin corrección, usada como referencia en la verificación.
    """
    X, _ = build_features(wide, terrain, models)
    tp = np.column_stack([
        wide[f"tp3__{m}"].to_numpy(dtype="float32") if f"tp3__{m}" in wide else np.full(len(wide), np.nan, "float32")
        for m in models])
    for k, m in enumerate(models):
        X[f"tp3__{m}"] = tp[:, k]
    avail = np.isfinite(tp).sum(axis=1)
    wet = (np.nan_to_num(tp, nan=0.0) >= RAIN_THRESHOLD_MM).sum(axis=1)
    X["tp3__mean"] = _nanmean(tp)
    with np.errstate(all="ignore"):
        X["tp3__max"] = np.where(avail > 0, np.nanmax(np.where(np.isfinite(tp), tp, -1), axis=1), np.nan)
    frac = np.where(avail > 0, wet / np.maximum(avail, 1), np.nan)
    X["wet_frac"] = frac
    return X, frac


__all__ = ["build_features", "build_rain_features", "TARGET_VAR", "VARIABLES", "LAPSE_RATE", "RAIN_THRESHOLD_MM"]
