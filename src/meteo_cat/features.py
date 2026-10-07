"""Construcción de la tabla de predictores, común a entrenamiento y operación.

Entrada "ancha": una fila por (lugar, hora válida, lead_day) con columnas `<var>__<modelo>`.
El modelo ML aprende el residuo respecto a una base física:
    base = media de modelos + gradiente térmico estándar * (altitud real - altitud a escala de modelo)
    objetivo = observación - base
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .terrain import TERRAIN_FEATURES

LAPSE_RATE = -0.0065  # °C/m, atmósfera estándar
TARGET_VAR = "temperature_2m"


def to_wide(long: pd.DataFrame, keys: list[str], variables) -> pd.DataFrame:
    """Pasa de tabla larga (una fila por modelo) a ancha con columnas `<var>__<modelo>`."""
    wide = long.pivot_table(index=keys, columns="model", values=list(variables), aggfunc="first")
    wide.columns = [f"{v}__{m}" for v, m in wide.columns]
    return wide.reset_index()


def model_columns(df: pd.DataFrame, var: str) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(var + "__"))


def build_features(wide: pd.DataFrame, terrain: dict[str, np.ndarray], models: tuple[str, ...],
                   variables) -> tuple[pd.DataFrame, np.ndarray]:
    """Devuelve (X, base). `terrain` contiene arrays alineados con las filas de `wide`."""
    X = pd.DataFrame(index=wide.index)
    for var in variables:
        for m in models:
            col = f"{var}__{m}"
            X[col] = wide[col].astype("float32") if col in wide else np.float32(np.nan)

    t_cols = [f"{TARGET_VAR}__{m}" for m in models]
    t = X[t_cols].to_numpy(dtype="float64")
    with np.errstate(all="ignore"):
        ens_mean = np.nanmean(t, axis=1)
        ens_std = np.nanstd(t, axis=1)
    X["ens_mean"] = ens_mean
    X["ens_std"] = ens_std
    X["n_models"] = np.isfinite(t).sum(axis=1)
    for var in variables:
        if var == TARGET_VAR:
            continue
        cols = [f"{var}__{m}" for m in models]
        with np.errstate(all="ignore"):
            X[f"{var}__mean"] = np.nanmean(X[cols].to_numpy(dtype="float64"), axis=1)

    time = pd.to_datetime(wide["time"], utc=True)
    hour = time.dt.hour.to_numpy() + time.dt.minute.to_numpy() / 60.0
    doy = time.dt.dayofyear.to_numpy()
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["lead_day"] = wide["lead_day"].to_numpy()

    for k in TERRAIN_FEATURES:
        X[k] = np.asarray(terrain[k], dtype="float32")
    dz = X["elevation"].to_numpy() - X["elev_smooth_10km"].to_numpy()
    X["elev_minus_smooth"] = dz

    base = ens_mean + LAPSE_RATE * dz
    return X, base


def feature_names(models: tuple[str, ...], variables) -> list[str]:
    """Orden de columnas de X para un conjunto de modelos y variables."""
    dummy = pd.DataFrame({"time": pd.to_datetime(["2024-01-01"], utc=True), "lead_day": [0]})
    terr = {k: np.zeros(1) for k in TERRAIN_FEATURES}
    X, _ = build_features(dummy, terr, models, variables)
    return list(X.columns)
