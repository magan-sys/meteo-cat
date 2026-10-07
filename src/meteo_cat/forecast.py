"""Ejecución operativa: previsión corregida en la rejilla de ~1 km, con nowcasting de 0 a 6 h."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Settings
from .features import TARGET_VAR, build_features, to_wide
from .grid import Grid, catalunya_mask, model_points
from .interp import LinearInterpolator, idw_residuals
from .sources.openmeteo import HISTORY_VARS, OpenMeteoClient
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


def _xy(lat, lon) -> np.ndarray:
    return np.column_stack([np.asarray(lat), np.asarray(lon) * np.cos(np.deg2rad(41.7))])


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.grid = Grid(settings.fine_grid_step)
        self.terrain = Terrain(settings.terrain_dir / f"terrain_{self.grid.step:g}.npz")
        self.mask = catalunya_mask(self.grid)
        lat2d, lon2d = self.grid.mesh()
        self.cell_lat, self.cell_lon = lat2d[self.mask], lon2d[self.mask]
        self.cell_terrain = {k: v[self.mask] for k, v in self.terrain.fields.items()}
        self.points = model_points(Grid(settings.model_grid_step))
        self.corrector = Corrector(settings.models_dir)
        if self.corrector.meta:
            self.models = tuple(self.corrector.meta["models"])
            self.variables = tuple(self.corrector.meta["variables"])
        else:
            log.warning("Sin modelo entrenado: se usa media de modelos + corrección por altitud")
            self.models, self.variables = settings.models, HISTORY_VARS
        self.cols = [f"{v}__{m}" for v in self.variables for m in self.models]

    def _predict_at(self, interp: LinearInterpolator, coarse: np.ndarray, time: pd.Timestamp,
                    lead_day: int, terrain: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        vals = interp(coarse)
        n = vals.shape[0]
        df = pd.DataFrame(vals, columns=self.cols)
        df["time"] = time
        df["lead_day"] = lead_day
        X, base = build_features(df, terrain, self.models, self.variables)
        return X["ens_mean"].to_numpy(), self.corrector.predict(X, base)[:n]

    def run(self, client: OpenMeteoClient, observations: Observations | None = None,
            now: datetime | None = None) -> dict:
        now = pd.Timestamp(now or datetime.now(timezone.utc)).floor("h")
        hours = self.settings.forecast_hours
        fc = client.forecast(self.points, self.models, hours, self.variables)
        wide = to_wide(fc, ["point", "time"], self.variables)
        for c in self.cols:
            if c not in wide:
                wide[c] = np.nan
        by_time = {t: g.set_index("point").reindex(range(len(self.points)))[self.cols].to_numpy()
                   for t, g in wide.groupby("time")}

        times = [t for t in sorted(by_time) if now <= t < now + pd.Timedelta(hours=hours)]
        interp = LinearInterpolator(_xy(self.points[:, 0], self.points[:, 1]), _xy(self.cell_lat, self.cell_lon))
        raw = np.full((len(times), self.mask.sum()), np.nan, dtype="float32")
        corr = np.full_like(raw, np.nan)
        for k, t in enumerate(times):
            lead = int(min(2, (t - now) // pd.Timedelta(hours=24)))
            raw[k], corr[k] = self._predict_at(interp, by_time[t], t, lead, self.cell_terrain)

        nowcast = None
        if observations is not None:
            nowcast = self._nowcast(observations, by_time, now, times, corr)

        return self._save(now, times, raw, corr, nowcast)

    def _nowcast(self, ob: Observations, by_time: dict, now, times, corr) -> dict | None:
        """Ajusta las primeras horas con el error actual en las estaciones."""
        obs = ob.obs[ob.obs["time"] <= now]
        counts = obs.groupby("time").size()
        counts = counts[counts >= NOWCAST_MIN_STATIONS]
        if counts.empty:
            log.info("Nowcasting: sin observaciones recientes suficientes")
            return None
        t0 = counts.index.max()
        if (now - t0) > pd.Timedelta(hours=NOWCAST_MAX_OBS_AGE_H) or t0 not in by_time:
            log.info("Nowcasting: última observación demasiado antigua (%s)", t0)
            return None
        latest = obs[obs["time"] == t0].merge(ob.stations, on="station")
        st_terrain = self.terrain.at(latest["lat"].to_numpy(), latest["lon"].to_numpy())
        alt = latest["altitude"].to_numpy(dtype="float64")
        st_terrain["elevation"] = np.where(np.isfinite(alt), alt, st_terrain["elevation"])
        interp_st = LinearInterpolator(_xy(self.points[:, 0], self.points[:, 1]),
                                       _xy(latest["lat"], latest["lon"]))
        _, pred_st = self._predict_at(interp_st, by_time[t0], t0, 0, st_terrain)
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

    def _save(self, now, times, raw, corr, nowcast) -> dict:
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
            "models": list(self.models),
            "ml_correction": self.corrector.meta is not None,
            "nowcast": nowcast,
        }
        (out / "latest.json").write_text(json.dumps(meta, indent=2))
        from .render import render_all
        render_all(full, times, self.grid, out)
        return meta
