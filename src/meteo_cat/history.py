"""Descarga del histórico (observaciones + previsiones pasadas) y montaje de la tabla de entrenamiento.

Todo se guarda en caché por meses en `<data_dir>/history`, de modo que la descarga se puede
interrumpir y reanudar (útil con los límites diarios de la API gratuita de Open-Meteo).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .features import to_wide
from .grid import catalunya_geometry
from .sources.openmeteo import HISTORY_VARS, OpenMeteoClient
from .sources.xema import XemaOpenData

log = logging.getLogger(__name__)

LEAD_DAYS = (0, 1, 2)


def _months(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur + timedelta(days=32)).replace(day=1)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


def load_stations(settings: Settings, refresh: bool = False) -> pd.DataFrame:
    path = settings.history_dir / "stations.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    st = XemaOpenData().stations()
    # Solo estaciones dentro (o muy cerca) de Catalunya.
    from shapely import contains_xy
    geom = catalunya_geometry().buffer(0.05)
    st = st[contains_xy(geom, st["lon"].to_numpy(), st["lat"].to_numpy())].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(path)
    return st


def download(settings: Settings, start: date, end: date) -> None:
    stations = load_stations(settings)
    log.info("%d estaciones", len(stations))
    obs_dir = settings.history_dir / "obs"
    fc_dir = settings.history_dir / "forecasts"
    obs_dir.mkdir(parents=True, exist_ok=True)
    fc_dir.mkdir(parents=True, exist_ok=True)
    xema = XemaOpenData()
    om = OpenMeteoClient(settings.open_meteo_api_key)
    points = stations[["lat", "lon"]].to_numpy()
    for a, b in _months(start, end):
        tag = a.strftime("%Y-%m")
        obs_path = obs_dir / f"{tag}.parquet"
        if not obs_path.exists():
            xema.hourly_observations(a, b).to_parquet(obs_path)
        fc_path = fc_dir / f"{tag}.parquet"
        if not fc_path.exists():
            df = om.previous_runs(points, settings.models, a, b, LEAD_DAYS, HISTORY_VARS)
            df["station"] = stations["station"].to_numpy()[df["point"].to_numpy()]
            df.drop(columns="point").to_parquet(fc_path)
        log.info("Mes %s listo", tag)


def assemble(settings: Settings) -> pd.DataFrame:
    """Une observaciones y previsiones: una fila por (station, time, lead_day)."""
    obs = _read_all(settings.history_dir / "obs")
    fc = _read_all(settings.history_dir / "forecasts")
    if obs.empty or fc.empty:
        raise RuntimeError("No hay histórico descargado. Ejecuta antes `meteo-cat history`.")
    wide = to_wide(fc, ["station", "time", "lead_day"], HISTORY_VARS)
    df = wide.merge(obs.rename(columns={"value": "obs"}), on=["station", "time"], how="inner")
    stations = load_stations(settings)
    return df.merge(stations[["station", "lat", "lon", "altitude"]], on="station", how="left")


def _read_all(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def station_terrain(df: pd.DataFrame, terrain) -> dict[str, np.ndarray]:
    """Terreno en la posición de cada fila. La altitud real de la estación sustituye a la media de celda."""
    t = terrain.at(df["lat"].to_numpy(), df["lon"].to_numpy())
    alt = df["altitude"].to_numpy(dtype="float64")
    t["elevation"] = np.where(np.isfinite(alt), alt, t["elevation"])
    return t
