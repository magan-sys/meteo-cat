"""Datos sintéticos para probar el flujo completo sin acceso a red."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from meteo_cat.config import Settings
from meteo_cat.grid import Grid
from meteo_cat.sources.nwp import RunFields
from meteo_cat.terrain import derive_features

MODELS = ("ecmwf_ifs", "gfs")


def synthetic_elevation(lat, lon):
    """Montaña al norte (Pirineo) y mar al sudeste."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    mount = 2500 * np.exp(-((lat - 42.5) ** 2) / 0.08) * (lon < 2.8)
    land = (lat - 40.5) * 1.2 + (2.6 - lon) > -0.4
    return np.where(land, 50 + mount + 300 * (2.6 - lon).clip(0), 0.0)


def _hours(time):
    t = pd.to_datetime(time, utc=True)
    return np.asarray(t.hour if hasattr(t, "hour") else t.dt.hour)


def truth(lat, lon, time):
    """Temperatura "real": ciclo diario + gradiente con la altitud + efecto de la latitud."""
    elev = synthetic_elevation(lat, lon)
    return 15 + 6 * np.sin(2 * np.pi * (_hours(time) - 9) / 24) - 0.0065 * elev - 2 * (lat - 41.5)


def model_value(model, lat, lon, time, lead_h, rng):
    """Los modelos ven el terreno suavizado y tienen sesgos distintos por modelo y horizonte."""
    smooth_elev = 0.5 * synthetic_elevation(lat, lon) + 400
    bias = {"ecmwf_ifs": 1.5, "gfs": -1.0}[model]
    amp = {"ecmwf_ifs": 4.0, "gfs": 7.0}[model]
    return (15 + amp * np.sin(2 * np.pi * (_hours(time) - 9) / 24) - 0.0065 * smooth_elev - 2 * (lat - 41.5)
            + bias + rng.normal(0, 0.6 + 0.4 * np.asarray(lead_h) / 24, np.shape(lat)))


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("METEOCAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("METEOCAT_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("METEOCAT_MODELS", ",".join(MODELS))
    monkeypatch.setenv("METEOCAT_FINE_GRID_STEP", "0.05")
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


class FakeNWP:
    """Sustituye a NWPClient: pasadas sintéticas en una rejilla de 0,25° sobre Catalunya."""

    def __init__(self, run: datetime, seed=0):
        self.run = run
        self.rng = np.random.default_rng(seed)
        self.lats = np.arange(43.75, 39.74, -0.25)
        self.lons = np.arange(-0.75, 4.26, 0.25)

    def latest_run(self, model, now=None, horizon_h=0):
        return self.run

    def fetch_run(self, model, run, max_step, variables=None):
        steps = np.arange(0, max_step + 1, 3)
        lat2d, lon2d = np.meshgrid(self.lats, self.lons, indexing="ij")
        t2m = np.stack([model_value(model, lat2d, lon2d, pd.Timestamp(run) + timedelta(hours=int(s)), s, self.rng)
                        for s in steps]).astype("float32")
        shape = t2m.shape
        data = {"t2m": t2m, "d2m": t2m - 5, "tcc": self.rng.uniform(0, 100, shape).astype("float32"),
                "u10": self.rng.normal(0, 3, shape).astype("float32"),
                "v10": self.rng.normal(0, 3, shape).astype("float32")}
        return RunFields(model, run, steps, self.lats, self.lons, data)
