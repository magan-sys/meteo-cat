"""Ejecución operativa: previsión corregida en la rejilla de ~1 km, con nowcasting de 0 a 6 h."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Settings
from .features import TARGET_VAR, build_features
from .grid import Grid, catalunya_mask
from .history import bilinear_weights, interp_steps, sample_points
from .interp import idw_residuals
from .sources.nwp import MODELS, VARIABLES, NWPClient
from .terrain import Terrain
from .train import Corrector

log = logging.getLogger(__name__)

NOWCAST_MAX_OBS_AGE_H = 3
NOWCAST_MIN_STATIONS = 10


@dataclass
class Observations:
    """Últimas observaciones: stations (station, lat, lon, altitude) y obs (station, time, value)."""
    stations: pd.DataFrame
    obs: pd.DataFrame


class ModelInputs:
    """Campos de todos los modelos interpolados en el tiempo a horas válidas concretas."""

    def __init__(self, runs: dict, times: list[pd.Timestamp]):
        self.runs = runs
        self.times = times
        self.fields = {}
        for m, run in runs.items():
            steps = [(t - pd.Timestamp(run.run)) / pd.Timedelta(hours=1) for t in times]
            self.fields[m] = interp_steps(run, steps)

    def at_points(self, k: int, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
        """Columnas `<var>__<modelo>` en los puntos para la hora k."""
        cols = {}
        for m, run in self.runs.items():
            w = bilinear_weights(run.lats, run.lons, lat, lon)
            for v in VARIABLES:
                cols[f"{v}__{m}"] = sample_points(self.fields[m][v][k], w)
        return pd.DataFrame(cols)


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.grid = Grid(settings.fine_grid_step)
        self.terrain = Terrain(settings.terrain_dir / f"terrain_{self.grid.step:g}.npz")
        self.mask = catalunya_mask(self.grid)
        lat2d, lon2d = self.grid.mesh()
        self.cell_lat, self.cell_lon = lat2d[self.mask], lon2d[self.mask]
        self.cell_terrain = {k: v[self.mask] for k, v in self.terrain.fields.items()}
        self.corrector = Corrector(settings.models_dir)
        if self.corrector.meta:
            self.models = tuple(self.corrector.meta["models"])
        else:
            log.warning("Sin modelo entrenado: se usa media de modelos + corrección por altitud")
            self.models = settings.models

    def fetch(self, client: NWPClient, now: pd.Timestamp) -> dict:
        runs = {}
        end = now + pd.Timedelta(hours=self.settings.forecast_hours)
        for m in self.models:
            run = client.latest_run(m, now.to_pydatetime(), self.settings.forecast_hours)
            if run is None:
                log.warning("%s: no hay ninguna pasada reciente publicada", m)
                continue
            spec = MODELS[m]
            max_step = math.ceil((end - pd.Timestamp(run)) / pd.Timedelta(hours=spec.step_h)) * spec.step_h
            fields = client.fetch_run(m, run, max_step)
            if fields is not None:
                runs[m] = fields
                log.info("%s: pasada %s, %d pasos", m, run.isoformat(), len(fields.steps))
        if not runs:
            raise RuntimeError("No se ha podido descargar ningún modelo")
        return runs

    def _predict(self, inputs: ModelInputs, k: int, lat, lon, terrain, lead_h: float):
        df = inputs.at_points(k, lat, lon)
        df["time"] = inputs.times[k]
        df["lead_h"] = lead_h
        X, base = build_features(df, terrain, self.models)
        return X["ens_mean"].to_numpy(), self.corrector.predict(X, base)

    def run(self, client: NWPClient, observations: Observations | None = None,
            now: datetime | None = None, runs: dict | None = None) -> dict:
        now = pd.Timestamp(now or datetime.now(timezone.utc)).floor("h")
        runs = runs or self.fetch(client, now)
        times = [now + pd.Timedelta(hours=h) for h in range(self.settings.forecast_hours)]
        inputs = ModelInputs(runs, times)
        ref_run = pd.Timestamp(runs[next(iter(runs))].run)

        raw = np.full((len(times), self.mask.sum()), np.nan, dtype="float32")
        corr = np.full_like(raw, np.nan)
        for k, t in enumerate(times):
            lead = (t - ref_run) / pd.Timedelta(hours=1)
            raw[k], corr[k] = self._predict(inputs, k, self.cell_lat, self.cell_lon, self.cell_terrain, lead)

        nowcast = None
        if observations is not None:
            nowcast = self._nowcast(observations, runs, now, times, ref_run, corr)
        return self._save(now, times, raw, corr, nowcast, runs)

    def _nowcast(self, ob: Observations, runs, now, times, ref_run, corr) -> dict | None:
        """Ajusta las primeras horas con el error actual en las estaciones."""
        obs = ob.obs[ob.obs["time"] <= now]
        counts = obs.groupby("time").size()
        counts = counts[counts >= NOWCAST_MIN_STATIONS]
        if counts.empty:
            log.info("Nowcasting: sin observaciones recientes suficientes")
            return None
        t0 = counts.index.max()
        if (now - t0) > pd.Timedelta(hours=NOWCAST_MAX_OBS_AGE_H):
            log.info("Nowcasting: última observación demasiado antigua (%s)", t0)
            return None
        latest = obs[obs["time"] == t0].merge(ob.stations, on="station")
        st_terrain = self.terrain.at(latest["lat"].to_numpy(), latest["lon"].to_numpy())
        alt = latest["altitude"].to_numpy(dtype="float64")
        st_terrain["elevation"] = np.where(np.isfinite(alt), alt, st_terrain["elevation"])
        at_t0 = ModelInputs(runs, [t0])
        lead0 = (t0 - ref_run) / pd.Timedelta(hours=1)
        _, pred_st = self._predict(at_t0, 0, latest["lat"].to_numpy(), latest["lon"].to_numpy(), st_terrain, lead0)
        resid = latest["value"].to_numpy() - pred_st
        r_grid = idw_residuals(latest["lat"], latest["lon"], st_terrain["elevation"], resid,
                               self.cell_lat, self.cell_lon, self.cell_terrain["elevation"])
        tau = self.corrector.tau_h
        for k, t in enumerate(times):
            h = (t - t0) / pd.Timedelta(hours=1)
            if h <= 4 * tau:
                corr[k] += (r_grid * np.exp(-h / tau)).astype("float32")
        log.info("Nowcasting con %d estaciones a %s (tau %.1f h)", len(latest), t0, tau)
        return {"obs_time": t0.isoformat(), "stations": int(len(latest)), "tau_h": tau,
                "mae_previo": float(np.nanmean(np.abs(resid)))}

    def _save(self, now, times, raw, corr, nowcast, runs) -> dict:
        out = self.settings.forecast_dir
        out.mkdir(parents=True, exist_ok=True)
        ny, nx = self.grid.shape
        full = np.full((len(times), ny, nx), np.nan, dtype="float32")
        full_raw = full.copy()
        full[:, self.mask] = corr
        full_raw[:, self.mask] = raw
        np.savez_compressed(out / "latest.npz", times=np.array([t.isoformat() for t in times]),
                            lats=self.grid.lats, lons=self.grid.lons, t2m=full, t2m_raw=full_raw)
        meta = {
            "issued": now.isoformat(),
            "variable": TARGET_VAR,
            "times": [t.isoformat() for t in times],
            "bounds": self.grid.bounds,
            "runs": {m: pd.Timestamp(r.run).isoformat() for m, r in runs.items()},
            "ml_correction": self.corrector.meta is not None,
            "trained_period": self.corrector.meta.get("period") if self.corrector.meta else None,
            "nowcast": nowcast,
        }
        (out / "latest.json").write_text(json.dumps(meta, indent=2))
        return meta
