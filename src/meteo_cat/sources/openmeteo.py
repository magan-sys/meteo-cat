"""Cliente de Open-Meteo para previsiones actuales y previsiones pasadas (entrenamiento).

- Previsión actual: /v1/forecast con varios modelos a la vez.
- Previsiones pasadas por horizonte: Previous Runs API (variables `<var>_previous_dayN`),
  que da para cada hora el valor que predijo el modelo N días antes. Es lo que permite
  aprender el error de cada modelo según el horizonte.

Se pide `elevation=nan` para desactivar la corrección de altitud de Open-Meteo: el ajuste
por orografía lo aprende nuestro modelo con el DEM propio.

Uso gratuito solo no comercial. Para uso comercial se define OPEN_METEO_API_KEY y se usan
los servidores "customer-" (las URL se pueden cambiar por variables de entorno).
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

FREE_HOSTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
}
CUSTOMER_HOSTS = {
    "forecast": "https://customer-api.open-meteo.com/v1/forecast",
    "previous_runs": "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
}
ENV_OVERRIDES = {"forecast": "OPEN_METEO_FORECAST_URL", "previous_runs": "OPEN_METEO_PREVIOUS_RUNS_URL"}

# Variables de modelo usadas como predictores.
FORECAST_VARS = ("temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m")
# En el histórico se piden menos variables para reducir el coste de la descarga.
HISTORY_VARS = ("temperature_2m", "cloud_cover", "wind_speed_10m")

MAX_POINTS_PER_REQUEST = 100


def endpoint(kind: str, api_key: str | None) -> str:
    override = os.environ.get(ENV_OVERRIDES[kind])
    if override:
        return override
    return (CUSTOMER_HOSTS if api_key else FREE_HOSTS)[kind]


class OpenMeteoClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None,
                 pause_s: float = 0.5, max_retries: int = 5):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.pause_s = pause_s
        self.max_retries = max_retries

    def _get(self, url: str, params: dict) -> list[dict]:
        if self.api_key:
            params = {**params, "apikey": self.api_key}
        delay = 10.0
        for attempt in range(self.max_retries):
            r = self.session.get(url, params=params, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("Open-Meteo %s, reintento en %.0f s", r.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Open-Meteo {r.status_code}: {r.text[:300]}")
            time.sleep(self.pause_s)
            data = r.json()
            return data if isinstance(data, list) else [data]
        raise RuntimeError("Open-Meteo: demasiados reintentos")

    def _points_params(self, lats, lons) -> dict:
        return {
            "latitude": ",".join(f"{x:.4f}" for x in lats),
            "longitude": ",".join(f"{x:.4f}" for x in lons),
            "elevation": ",".join("nan" for _ in lats),
            "timezone": "GMT",
        }

    def forecast(self, points: np.ndarray, models: tuple[str, ...], hours: int,
                 variables: tuple[str, ...] = FORECAST_VARS) -> pd.DataFrame:
        """Previsión actual. Devuelve tabla larga: point, time, model, <variables>."""
        url = endpoint("forecast", self.api_key)
        frames = []
        for start in range(0, len(points), MAX_POINTS_PER_REQUEST):
            chunk = points[start:start + MAX_POINTS_PER_REQUEST]
            params = {
                **self._points_params(chunk[:, 0], chunk[:, 1]),
                "hourly": ",".join(variables),
                "models": ",".join(models),
                "forecast_hours": hours,
                "past_hours": 6,
            }
            for k, loc in enumerate(self._get(url, params)):
                frames.append(parse_hourly(loc["hourly"], variables, models, point=start + k))
        return pd.concat(frames, ignore_index=True)

    def previous_runs(self, points: np.ndarray, models: tuple[str, ...], start: date, end: date,
                      lead_days: tuple[int, ...] = (0, 1, 2),
                      variables: tuple[str, ...] = HISTORY_VARS) -> pd.DataFrame:
        """Previsiones pasadas por horizonte. Tabla larga: point, time, model, lead_day, <variables>."""
        url = endpoint("previous_runs", self.api_key)
        req_vars = [v if d == 0 else f"{v}_previous_day{d}" for d in lead_days for v in variables]
        frames = []
        for s in range(0, len(points), MAX_POINTS_PER_REQUEST):
            chunk = points[s:s + MAX_POINTS_PER_REQUEST]
            params = {
                **self._points_params(chunk[:, 0], chunk[:, 1]),
                "hourly": ",".join(req_vars),
                "models": ",".join(models),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
            for k, loc in enumerate(self._get(url, params)):
                frames.append(parse_previous_runs(loc["hourly"], variables, models, lead_days, point=s + k))
        return pd.concat(frames, ignore_index=True)


def _split_model(key: str, models: tuple[str, ...]) -> tuple[str, str | None]:
    """Separa el sufijo de modelo de una clave de Open-Meteo ('temperature_2m_icon_eu')."""
    for m in sorted(models, key=len, reverse=True):
        if key.endswith("_" + m):
            return key[: -len(m) - 1], m
    return key, None


def parse_hourly(hourly: dict, variables, models, point: int) -> pd.DataFrame:
    times = pd.to_datetime(hourly["time"], utc=True)
    cols: dict[str, dict[str, np.ndarray]] = {m: {} for m in models}
    for key, values in hourly.items():
        if key == "time":
            continue
        base, model = _split_model(key, models)
        if model is None and len(models) == 1:
            model = models[0]
        if model in cols and base in variables:
            cols[model][base] = np.asarray(values, dtype="float64")
    frames = []
    for m, vals in cols.items():
        if not vals:
            continue
        df = pd.DataFrame({"time": times, **vals})
        df.insert(0, "model", m)
        df.insert(0, "point", point)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


_PREV_RE = re.compile(r"^(?P<var>.+?)(?:_previous_day(?P<day>\d+))?$")


def parse_previous_runs(hourly: dict, variables, models, lead_days, point: int) -> pd.DataFrame:
    times = pd.to_datetime(hourly["time"], utc=True)
    cols: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, values in hourly.items():
        if key == "time":
            continue
        base, model = _split_model(key, models)
        if model is None and len(models) == 1:
            model = models[0]
        m = _PREV_RE.match(base)
        if model is None or not m or m["var"] not in variables:
            continue
        day = int(m["day"] or 0)
        if day not in lead_days:
            continue
        cols.setdefault((model, day), {})[m["var"]] = np.asarray(values, dtype="float64")
    frames = []
    for (model, day), vals in sorted(cols.items()):
        df = pd.DataFrame({"time": times, **vals})
        df.insert(0, "lead_day", day)
        df.insert(0, "model", model)
        df.insert(0, "point", point)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
