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

# Códigos de variable XEMA (conjunto de metadatos de variables 4fb2-n3yi).
VAR_TEMPERATURE = "32"    # Temperatura, °C
VAR_PRECIPITATION = "35"  # Precipitació, mm por periodo semihorario

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
        """Lecturas en hora en punto: station, time (UTC), value.

        Se consulta día a día: las consultas de un mes entero con paginación son muy lentas en el portal.
        """
        frames = []
        day = start
        while day <= end:
            nxt = day + timedelta(days=1)
            where = (f"codi_variable='{variable}' AND data_lectura >= '{day.isoformat()}T00:00:00' "
                     f"AND data_lectura < '{nxt.isoformat()}T00:00:00'")
            rows = self._query(READINGS_DATASET, {
                "$select": "codi_estacio,data_lectura,valor_lectura",
                "$where": where,
            })
            if rows:
                frames.append(_to_hourly(pd.DataFrame(rows)))
            day = nxt
        log.info("XEMA %s a %s: %d lecturas horarias", start, end, sum(len(f) for f in frames))
        if not frames:
            return pd.DataFrame(columns=["station", "time", "value"])
        return pd.concat(frames, ignore_index=True)

    def precipitation_3h(self, start: date, end: date) -> pd.DataFrame:
        """Lluvia acumulada en las 3 h que terminan en cada hora múltiplo de 3 UTC: station, time, ppt3.

        Se asume que cada lectura semihoraria cubre los 30 minutos que empiezan en `data_lectura`;
        la ventana (t-3h, t] suma las lecturas de t-3h a t-30min. Solo se dan ventanas completas
        (las 6 lecturas presentes).
        """
        frames = []
        day = start - timedelta(days=1)  # para completar la primera ventana del día
        while day <= end:
            nxt = day + timedelta(days=1)
            where = (f"codi_variable='{VAR_PRECIPITATION}' AND data_lectura >= '{day.isoformat()}T00:00:00' "
                     f"AND data_lectura < '{nxt.isoformat()}T00:00:00'")
            rows = self._query(READINGS_DATASET, {"$select": "codi_estacio,data_lectura,valor_lectura",
                                                  "$where": where})
            if rows:
                frames.append(pd.DataFrame(rows))
            day = nxt
        if not frames:
            return pd.DataFrame(columns=["station", "time", "ppt3"])
        return sum_3h(pd.concat(frames, ignore_index=True), start, end)


def sum_3h(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    t = pd.to_datetime(df["data_lectura"]).dt.tz_localize("UTC")
    v = pd.to_numeric(df["valor_lectura"], errors="coerce")
    d = pd.DataFrame({"station": df["codi_estacio"], "t": t, "v": v})
    d = d[d["v"].between(0, 200)].drop_duplicates(["station", "t"])
    # Ventana que termina en el siguiente múltiplo de 3 h tras el final de la lectura.
    end_t = (d["t"] + pd.Timedelta(minutes=30)).dt.ceil("3h")
    g = d.assign(time=end_t).groupby(["station", "time"])["v"].agg(["sum", "count"]).reset_index()
    g = g[g["count"] == 6]
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    g = g[(g["time"] > lo) & (g["time"] <= hi)]
    return g.rename(columns={"sum": "ppt3"})[["station", "time", "ppt3"]].reset_index(drop=True)


def _to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["data_lectura"]).dt.tz_localize("UTC")
    out = pd.DataFrame({"station": df["codi_estacio"], "time": t,
                        "value": pd.to_numeric(df["valor_lectura"], errors="coerce")})
    # Control de calidad mínimo: rango físico plausible en Catalunya.
    out = out[(out["time"].dt.minute == 0) & out["value"].between(-40, 50)]
    return out.drop_duplicates(["station", "time"]).reset_index(drop=True)
