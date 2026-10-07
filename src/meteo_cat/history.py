"""Histórico para entrenar: observaciones XEMA y previsiones pasadas de los modelos en cada estación.

Se usan las pasadas de las 00 UTC de cada día, con pasos cada 3 h hasta 72 h. Todo se guarda
en caché por meses en `<data_dir>/history`, de modo que la descarga se puede interrumpir y
reanudar.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .grid import catalunya_geometry
from .sources.nwp import PRECIP, VARIABLES, NWPClient, RunFields
from .sources.xema import XemaOpenData

log = logging.getLogger(__name__)

LEAD_STEPS_H = tuple(range(0, 73, 3))


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
    from shapely import contains_xy
    geom = catalunya_geometry().buffer(0.05)
    st = st[contains_xy(geom, st["lon"].to_numpy(), st["lat"].to_numpy())].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(path)
    return st


def bilinear_weights(lats: np.ndarray, lons: np.ndarray, plat: np.ndarray, plon: np.ndarray):
    """Índices y pesos para interpolar una rejilla regular (lats de norte a sur) a puntos."""
    dlat = lats[0] - lats[1]
    dlon = lons[1] - lons[0]
    fi = np.clip((lats[0] - plat) / dlat, 0, len(lats) - 1.000001)
    fj = np.clip((plon - lons[0]) / dlon, 0, len(lons) - 1.000001)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    wi, wj = fi - i0, fj - j0
    return i0, j0, wi, wj


def sample_points(field: np.ndarray, w) -> np.ndarray:
    """field (..., lat, lon) -> (..., n_puntos)."""
    i0, j0, wi, wj = w
    return ((1 - wi) * (1 - wj) * field[..., i0, j0] + (1 - wi) * wj * field[..., i0, j0 + 1]
            + wi * (1 - wj) * field[..., i0 + 1, j0] + wi * wj * field[..., i0 + 1, j0 + 1])


def interp_steps(run: RunFields, target_steps) -> dict[str, np.ndarray]:
    """Interpola en el tiempo los campos de una pasada a los pasos pedidos (NaN fuera de rango)."""
    out = {}
    target = np.asarray(target_steps, dtype=float)
    for v, arr in run.data.items():
        res = np.full((len(target),) + arr.shape[1:], np.nan, dtype="float32")
        for k, s in enumerate(target):
            hi = np.searchsorted(run.steps, s)
            if hi < len(run.steps) and run.steps[hi] == s:
                res[k] = arr[hi]
            elif 0 < hi < len(run.steps):
                a = (s - run.steps[hi - 1]) / (run.steps[hi] - run.steps[hi - 1])
                res[k] = (1 - a) * arr[hi - 1] + a * arr[hi]
        out[v] = res
    return out


def run_at_points(run: RunFields, plat, plon, steps=LEAD_STEPS_H) -> pd.DataFrame:
    """Valores de una pasada en puntos: filas (punto, paso) con columnas VARIABLES."""
    w = bilinear_weights(run.lats, run.lons, np.asarray(plat), np.asarray(plon))
    fields = interp_steps(run, steps)
    n = len(plat)
    df = pd.DataFrame({
        "point": np.tile(np.arange(n), len(steps)),
        "lead_h": np.repeat(np.asarray(steps), n),
    })
    df["time"] = pd.Timestamp(run.run) + pd.to_timedelta(df["lead_h"], unit="h")
    for v in VARIABLES:
        df[v] = sample_points(fields[v], w).reshape(-1).astype("float32")
    return df


def run_tp3_at_points(run: RunFields, plat, plon, steps=LEAD_STEPS_H) -> pd.DataFrame:
    """Lluvia de las 3 h previas a cada paso (mm), en puntos: filas (punto, paso) con columna tp3."""
    w = bilinear_weights(run.lats, run.lons, np.asarray(plat), np.asarray(plon))
    steps = [s for s in steps if s >= 3]
    acc = sample_points(interp_steps(run, steps)[PRECIP], w)
    acc_prev = sample_points(interp_steps(run, [s - 3 for s in steps])[PRECIP], w)
    n = len(plat)
    df = pd.DataFrame({"point": np.tile(np.arange(n), len(steps)), "lead_h": np.repeat(np.asarray(steps), n)})
    df["time"] = pd.Timestamp(run.run) + pd.to_timedelta(df["lead_h"], unit="h")
    df["tp3"] = np.clip(acc - acc_prev, 0, None).reshape(-1).astype("float32")
    return df


def download(settings: Settings, start: date, end: date, what=("temp", "precip")) -> None:
    stations = load_stations(settings)
    log.info("%d estaciones", len(stations))
    xema = XemaOpenData()
    client = NWPClient()
    jobs = []
    if "temp" in what:
        jobs.append(("obs", "nwp", xema.hourly_observations, VARIABLES, run_at_points))
    if "precip" in what:
        jobs.append(("obs_ppt", "nwp_tp", xema.precipitation_3h, (PRECIP,), run_tp3_at_points))
    for a, b in _months(start, end):
        # Un mes incompleto se guarda como ".partial" y se vuelve a descargar la próxima vez.
        complete = a.day == 1 and (b + timedelta(days=1)).day == 1
        tag = a.strftime("%Y-%m") + ("" if complete else ".partial")
        for obs_name, nwp_name, get_obs, variables, at_points in jobs:
            _download_month(settings, client, stations, a, b, tag, complete,
                            settings.history_dir / obs_name, settings.history_dir / nwp_name,
                            get_obs, variables, at_points)


def _download_month(settings, client, stations, a, b, tag, complete, obs_dir, nwp_dir,
                    get_obs, variables, at_points) -> None:
    obs_dir.mkdir(parents=True, exist_ok=True)
    obs_path = obs_dir / f"{tag}.parquet"
    if not obs_path.exists() or not complete:
        get_obs(a, b).to_parquet(obs_path)
        if complete:
            (obs_dir / f"{tag}.partial.parquet").unlink(missing_ok=True)
    for model in settings.models:
        path = nwp_dir / model / f"{tag}.parquet"
        if path.exists() and complete:
            continue
        frames = []
        day = a
        while day <= b:
            run = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            try:
                fields = client.fetch_run(model, run, max(LEAD_STEPS_H), variables)
            except Exception as e:  # una pasada que falla no detiene el histórico
                log.warning("%s %s: %s", model, run, e)
                fields = None
            if fields is not None:
                df = at_points(fields, stations["lat"].to_numpy(), stations["lon"].to_numpy())
                df["station"] = stations["station"].to_numpy()[df.pop("point").to_numpy()]
                frames.append(df)
            day += timedelta(days=1)
        if frames:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_parquet(path)
            if complete:
                (path.parent / f"{tag}.partial.parquet").unlink(missing_ok=True)
        log.info("%s %s %s: %d pasadas", nwp_dir.name, model, tag, len(frames))


def assemble(settings: Settings) -> pd.DataFrame:
    """Une observaciones y modelos: una fila por (station, time, lead_h) con columnas `<var>__<modelo>`."""
    obs = _read_all(settings.history_dir / "obs")
    if obs.empty:
        raise RuntimeError("No hay histórico descargado. Ejecuta antes `meteo-cat history`.")
    wide = None
    for model in settings.models:
        df = _read_all(settings.history_dir / "nwp" / model)
        if df.empty:
            log.warning("Sin histórico para %s", model)
            continue
        df = df.rename(columns={v: f"{v}__{model}" for v in VARIABLES})
        keys = ["station", "time", "lead_h"]
        wide = df if wide is None else wide.merge(df, on=keys, how="outer")
    if wide is None:
        raise RuntimeError("No hay histórico de modelos descargado.")
    df = wide.merge(obs.rename(columns={"value": "obs"}), on=["station", "time"], how="inner")
    stations = load_stations(settings)
    return df.merge(stations[["station", "lat", "lon", "altitude"]], on="station", how="left")


def assemble_precip(settings: Settings) -> pd.DataFrame:
    """Como `assemble`, pero con la lluvia observada en 3 h (`obs`) y la de cada modelo (`tp3__<modelo>`)."""
    obs = _read_all(settings.history_dir / "obs_ppt")
    if obs.empty:
        raise RuntimeError("No hay histórico de lluvia. Ejecuta antes `meteo-cat history`.")
    keys = ["station", "time", "lead_h"]
    wide = None
    for model in settings.models:
        temp = _read_all(settings.history_dir / "nwp" / model)
        rain = _read_all(settings.history_dir / "nwp_tp" / model)
        if temp.empty or rain.empty:
            log.warning("Sin histórico de lluvia para %s", model)
            continue
        df = temp.merge(rain, on=keys, how="inner")
        df = df.rename(columns={v: f"{v}__{model}" for v in VARIABLES + ("tp3",)})
        wide = df if wide is None else wide.merge(df, on=keys, how="outer")
    if wide is None:
        raise RuntimeError("No hay histórico de lluvia de los modelos.")
    df = wide.merge(obs.rename(columns={"ppt3": "obs"}), on=["station", "time"], how="inner")
    stations = load_stations(settings)
    return df.merge(stations[["station", "lat", "lon", "altitude"]], on="station", how="left")


def _read_all(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet"))
    # Si existe el mes completo, se ignora su versión parcial.
    full = {f.name.split(".")[0] for f in files if ".partial" not in f.name}
    files = [f for f in files if ".partial" not in f.name or f.name.split(".")[0] not in full]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def station_terrain(df: pd.DataFrame, terrain) -> dict[str, np.ndarray]:
    """Terreno en la posición de cada fila. La altitud real de la estación sustituye a la media de celda."""
    t = terrain.at(df["lat"].to_numpy(), df["lon"].to_numpy())
    alt = df["altitude"].to_numpy(dtype="float64")
    t["elevation"] = np.where(np.isfinite(alt), alt, t["elevation"])
    return t
