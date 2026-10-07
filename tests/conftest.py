"""Datos sintéticos para probar el flujo completo sin acceso a red."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteo_cat.config import Settings
from meteo_cat.grid import Grid
from meteo_cat.terrain import derive_features

MODELS = ("model_a", "model_b")


def synthetic_elevation(lat, lon):
    """Montaña al norte (Pirineo) y mar al sudeste."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    mount = 2500 * np.exp(-((lat - 42.5) ** 2) / 0.08) * (lon < 2.8)
    land = (lat - 40.5) * 1.2 + (2.6 - lon) > -0.4
    return np.where(land, 50 + mount + 300 * (2.6 - lon).clip(0), 0.0)


def truth(lat, lon, time):
    """Temperatura "real": ciclo diario + gradiente con la altitud + efecto del mar."""
    t = pd.to_datetime(time, utc=True)
    hour = np.asarray(t.hour if hasattr(t, "hour") else t.dt.hour)
    elev = synthetic_elevation(lat, lon)
    return 15 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) - 0.0065 * elev - 2 * (lat - 41.5)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("METEOCAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("METEOCAT_MODELS", ",".join(MODELS))
    monkeypatch.setenv("METEOCAT_FINE_GRID_STEP", "0.05")
    monkeypatch.setenv("METEOCAT_MODEL_GRID_STEP", "0.2")
    monkeypatch.setenv("METEOCAT_FORECAST_HOURS", "24")
    s = Settings()
    grid = Grid(s.fine_grid_step)
    lat2d, lon2d = grid.mesh()
    elev = synthetic_elevation(lat2d, lon2d)
    land = (elev > 0).astype(float)
    feats = derive_features(elev, elev**2, land, grid)
    s.terrain_dir.mkdir(parents=True)
    np.savez_compressed(s.terrain_dir / f"terrain_{grid.step:g}.npz", lats=grid.lats, lons=grid.lons,
                        **{k: v.astype("float32") for k, v in feats.items()})
    return s


def model_value(model, lat, lon, time, lead_day, rng):
    """Los modelos ven el terreno suavizado y tienen sesgos distintos por modelo y horizonte."""
    t = pd.to_datetime(time, utc=True)
    hour = np.asarray(t.hour if hasattr(t, "hour") else t.dt.hour)
    smooth_elev = 0.5 * synthetic_elevation(lat, lon) + 400
    bias = {"model_a": 1.5, "model_b": -1.0}[model]
    amp = {"model_a": 4.0, "model_b": 7.0}[model]
    return (15 + amp * np.sin(2 * np.pi * (hour - 9) / 24) - 0.0065 * smooth_elev - 2 * (lat - 41.5)
            + bias + rng.normal(0, 0.6 + 0.4 * lead_day, np.shape(lat)))


class FakeOpenMeteo:
    """Sustituye al cliente HTTP en el flujo operativo."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def forecast(self, points, models, hours, variables):
        times = pd.date_range("2026-10-07T00:00Z", periods=hours + 6, freq="h")
        rows = []
        for m in models:
            for t in times:
                df = pd.DataFrame({"point": np.arange(len(points)), "time": t, "model": m})
                for v in variables:
                    df[v] = (model_value(m, points[:, 0], points[:, 1], t, 0, self.rng)
                             if v == "temperature_2m" else self.rng.uniform(0, 100, len(points)))
                rows.append(df)
        return pd.concat(rows, ignore_index=True)
