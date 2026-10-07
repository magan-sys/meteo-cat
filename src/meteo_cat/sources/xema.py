"""Observaciones de la Xarxa d'Estacions Meteorològiques Automàtiques (XEMA) de Meteocat.

Fuente actual: portal de datos abiertos de la Generalitat (Socrata), sin clave:
  - Metadatos de estaciones: conjunto yqwd-vj5e
  - Lecturas semihorarias:   conjunto nzvn-apee
Un token de aplicación Socrata (SOCRATA_APP_TOKEN) es opcional y amplía los límites.

Cuando esté disponible la clave de la API XEMA de Meteocat (METEOCAT_API_KEY) se añadirá
un segundo adaptador para datos en tiempo real, necesario para el nowcasting operativo.

Pendiente de verificar con acceso real a los datos: nombres de campos y zona horaria de
`data_lectura`. `meteo_cat.verify.time_alignment` comprueba el desfase horario contra los
modelos durante el entrenamiento.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://analisi.transparenciacatalunya.cat/resource"
STATIONS_DATASET = "yqwd-vj5e"
READINGS_DATASET = "nzvn-apee"

# Códigos de variable XEMA.
VAR_TEMPERATURE = "32"

PAGE = 50_000


class XemaOpenData:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        token = os.environ.get("SOCRATA_APP_TOKEN")
        if token:
            self.session.headers["X-App-Token"] = token

    def _query(self, dataset: str, params: dict) -> list[dict]:
        rows, offset = [], 0
        while True:
            r = self.session.get(f"{BASE}/{dataset}.json",
                                 params={**params, "$limit": PAGE, "$offset": offset}, timeout=300)
            r.raise_for_status()
            page = r.json()
            rows.extend(page)
            if len(page) < PAGE:
                return rows
            offset += PAGE

    def stations(self) -> pd.DataFrame:
        """Estaciones: station, name, lat, lon, altitude, active."""
        rows = self._query(STATIONS_DATASET, {})
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "station": df["codi_estacio"],
            "name": df.get("nom_estacio"),
            "lat": pd.to_numeric(df["latitud"]),
            "lon": pd.to_numeric(df["longitud"]),
            "altitude": pd.to_numeric(df.get("altitud"), errors="coerce"),
        })
        estado = df.get("nom_estat_ema")
        out["active"] = estado.str.contains("Operativa", case=False, na=False) if estado is not None else True
        return out

    def hourly_observations(self, start: date, end: date, variable: str = VAR_TEMPERATURE) -> pd.DataFrame:
        """Lecturas en hora en punto: station, time (UTC), value. Descarga mes a mes."""
        frames = []
        cur = start
        while cur <= end:
            nxt = min((cur.replace(day=1) + timedelta(days=32)).replace(day=1), end + timedelta(days=1))
            where = (f"codi_variable='{variable}' AND data_lectura >= '{cur.isoformat()}T00:00:00' "
                     f"AND data_lectura < '{nxt.isoformat()}T00:00:00'")
            rows = self._query(READINGS_DATASET, {
                "$select": "codi_estacio,data_lectura,valor_lectura",
                "$where": where,
                "$order": "data_lectura",
            })
            log.info("XEMA %s: %d lecturas", cur.strftime("%Y-%m"), len(rows))
            if rows:
                frames.append(_to_hourly(pd.DataFrame(rows)))
            cur = nxt
        if not frames:
            return pd.DataFrame(columns=["station", "time", "value"])
        return pd.concat(frames, ignore_index=True)


def _to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["data_lectura"]).dt.tz_localize("UTC")
    out = pd.DataFrame({"station": df["codi_estacio"], "time": t,
                        "value": pd.to_numeric(df["valor_lectura"], errors="coerce")})
    # Control de calidad mínimo: rango físico plausible en Catalunya.
    out = out[(out["time"].dt.minute == 0) & out["value"].between(-40, 50)]
    return out.drop_duplicates(["station", "time"]).reset_index(drop=True)
