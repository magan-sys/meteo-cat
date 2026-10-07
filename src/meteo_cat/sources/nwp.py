"""Descarga directa de modelos numéricos abiertos (sin claves ni límites de peticiones).

- ECMWF IFS 0,25° y ECMWF AIFS (modelo de IA) desde la réplica pública de ECMWF Open Data
  en Google Cloud. Histórico desde 2023 (IFS) y 2024 (AIFS).
- NOAA GFS 0,25° desde el bucket público de AWS. Histórico desde 2021.

Cada fichero GRIB2 es global (~100 MB por paso), pero viene con un índice que permite
descargar solo los mensajes necesarios con peticiones HTTP parciales (~0,6 MB por campo).
Los campos se recortan a la zona de Catalunya nada más leerlos.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

log = logging.getLogger(__name__)

# Recorte (con margen para interpolar en el borde de Catalunya).
CROP_LAT = (39.75, 43.75)
CROP_LON = (-0.75, 4.25)

# Variables comunes a todos los modelos, en unidades de salida.
VARIABLES = ("t2m", "d2m", "tcc", "u10", "v10")  # °C, °C, %, m/s, m/s
# Precipitación acumulada desde el inicio de la pasada (mm). Se descarga aparte porque su
# histórico se monta por separado.
PRECIP = "tp"

ECMWF_BASE = "https://storage.googleapis.com/ecmwf-open-data"
GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

ECMWF_PARAMS = {"t2m": "2t", "d2m": "2d", "tcc": "tcc", "u10": "10u", "v10": "10v", "tp": "tp"}
GFS_PARAMS = {
    "t2m": ":TMP:2 m above ground:",
    "d2m": ":DPT:2 m above ground:",
    "tcc": ":TCDC:entire atmosphere:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
    "tp": ":APCP:surface:0-",  # acumulado desde el inicio ("0-N hour acc fcst")
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str        # "ecmwf" | "gfs"
    path: str = ""     # subcarpeta en ECMWF Open Data
    step_h: int = 3    # resolución temporal usada
    delay_h: float = 6  # horas tras la pasada hasta que suele estar publicada


MODELS = {
    "ecmwf_ifs": ModelSpec("ecmwf_ifs", "ecmwf", "ifs/0p25/oper", 3, 8),
    "ecmwf_aifs": ModelSpec("ecmwf_aifs", "ecmwf", "aifs-single/0p25/oper", 6, 7),
    "gfs": ModelSpec("gfs", "gfs", "", 3, 5),
}


@dataclass
class RunFields:
    """Campos recortados de una pasada: data[var] tiene forma (pasos, lat, lon)."""
    model: str
    run: datetime
    steps: np.ndarray
    lats: np.ndarray   # de norte a sur
    lons: np.ndarray   # de oeste a este
    data: dict[str, np.ndarray]


class NWPClient:
    def __init__(self, session: requests.Session | None = None, workers: int = 8, retries: int = 4):
        self.session = session or requests.Session()
        self.workers = workers
        self.retries = retries

    def _get(self, url: str, rng: tuple[int, int] | None = None) -> bytes | None:
        headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
        delay = 2.0
        for _ in range(self.retries):
            try:
                r = self.session.get(url, headers=headers, timeout=120)
            except requests.RequestException as e:
                log.warning("%s: %s", url, e)
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504) or b"<Code>SlowDown" in r.content[:200]:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.content
        raise RuntimeError(f"No se pudo descargar {url}")

    # ---- localización de los mensajes en cada fichero ----

    def _ecmwf_ranges(self, spec: ModelSpec, run: datetime, step: int, variables) -> tuple[str, dict] | None:
        stem = f"{run:%Y%m%d}/{run:%H}z/{spec.path}/{run:%Y%m%d%H}0000-{step}h-oper-fc"
        idx = self._get(f"{ECMWF_BASE}/{stem}.index")
        if idx is None:
            return None
        want = {ECMWF_PARAMS[v]: v for v in variables}
        ranges = {}
        for line in idx.decode().splitlines():
            d = json.loads(line)
            if d.get("levtype") == "sfc" and d.get("param") in want:
                ranges[want[d["param"]]] = (d["_offset"], d["_offset"] + d["_length"] - 1)
        return f"{ECMWF_BASE}/{stem}.grib2", ranges

    def _gfs_ranges(self, run: datetime, step: int, variables) -> tuple[str, dict] | None:
        url = f"{GFS_BASE}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}"
        idx = self._get(url + ".idx")
        if idx is None:
            return None
        lines = idx.decode().splitlines()
        offsets = [int(line.split(":")[1]) for line in lines]
        ranges = {}
        for k, line in enumerate(lines):
            for v in variables:
                if v in ranges or GFS_PARAMS[v] not in line:
                    continue
                if v == "tp":
                    ok = " acc fcst:" in line  # "0-N hour acc fcst" o "0-1 day acc fcst"
                else:
                    # Solo el valor instantáneo ("N hour fcst" o "anl"), no medias ni acumulados.
                    ok = (" hour fcst:" in line or ":anl:" in line) and "-" not in line.split(":")[5]
                if ok:
                    end = offsets[k + 1] - 1 if k + 1 < len(offsets) else ""
                    ranges[v] = (offsets[k], end)
        return url, ranges

    # ---- descarga y decodificación ----

    def fetch_run(self, model: str, run: datetime, max_step: int, variables=VARIABLES) -> RunFields | None:
        spec = MODELS[model]
        steps = list(range(0, max_step + 1, spec.step_h))

        def locate(step):
            if spec.source == "ecmwf":
                return step, self._ecmwf_ranges(spec, run, step, variables)
            return step, self._gfs_ranges(run, step, variables)

        with ThreadPoolExecutor(self.workers) as ex:
            located = [x for x in ex.map(locate, steps) if x[1] is not None]
        if not located:
            return None
        jobs = [(step, v, url, rng) for step, (url, ranges) in located for v, rng in ranges.items()]

        def download(job):
            step, v, url, rng = job
            return step, v, self._get(url, (rng[0], rng[1]) if rng[1] != "" else None)

        with ThreadPoolExecutor(self.workers) as ex:
            blobs = list(ex.map(download, jobs))

        got_steps = sorted({s for s, _ in located})
        lats = lons = None
        data = {v: None for v in variables}
        for step, v, blob in blobs:
            if blob is None:
                continue
            la, lo, field, units = decode_crop(blob)
            if lats is None:
                lats, lons = la, lo
                data = {var: np.full((len(got_steps),) + field.shape, np.nan, dtype="float32") for var in variables}
            data[v][got_steps.index(step)] = convert_units(v, field, spec.source, units)
        if lats is None:
            return None
        if PRECIP in data and got_steps[0] == 0:
            data[PRECIP][0] = 0.0  # al inicio de la pasada no hay acumulado (y a veces no hay campo)
        return RunFields(model, run, np.array(got_steps), lats, lons, data)

    def latest_run(self, model: str, now: datetime | None = None, horizon_h: int = 0,
                   lookback_h: int = 30) -> datetime | None:
        """Pasada más reciente ya publicada entera hasta `now + horizon_h`.

        Las pasadas se publican paso a paso a lo largo de una o dos horas: se comprueba que
        exista el índice del primer paso y del último paso necesario.
        """
        spec = MODELS[model]
        now = now or datetime.now(timezone.utc)
        t = now.replace(minute=0, second=0, microsecond=0)
        t -= timedelta(hours=t.hour % 6)
        for _ in range(lookback_h // 6 + 1):
            need = (now - t).total_seconds() / 3600 + horizon_h
            last = int(-(-need // spec.step_h) * spec.step_h)
            if all(self._has_step(spec, t, s) for s in (0, last)):
                return t
            t -= timedelta(hours=6)
        return None

    def _has_step(self, spec: ModelSpec, run: datetime, step: int) -> bool:
        found = (self._ecmwf_ranges(spec, run, step, ("t2m",)) if spec.source == "ecmwf"
                 else self._gfs_ranges(run, step, ("t2m",)))
        return bool(found and found[1])

def decode_crop(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Decodifica un mensaje GRIB2 en rejilla regular lat/lon, lo recorta a CROP_LAT/CROP_LON
    y devuelve también sus unidades."""
    import eccodes

    h = eccodes.codes_new_from_message(blob)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        scans_neg = eccodes.codes_get(h, "jScansPositively") == 0
        vals = eccodes.codes_get_values(h).reshape(nj, ni)
        units = eccodes.codes_get(h, "units")
    finally:
        eccodes.codes_release(h)
    lats = lat1 - dlat * np.arange(nj) if scans_neg else lat1 + dlat * np.arange(nj)
    lons = (lon1 + dlon * np.arange(ni) + 180.0) % 360.0 - 180.0
    order = np.argsort(lons)
    lons, vals = lons[order], vals[:, order]
    if not scans_neg:
        lats, vals = lats[::-1], vals[::-1]
    i = (lats >= CROP_LAT[0] - 1e-6) & (lats <= CROP_LAT[1] + 1e-6)
    j = (lons >= CROP_LON[0] - 1e-6) & (lons <= CROP_LON[1] + 1e-6)
    return np.round(lats[i], 4), np.round(lons[j], 4), vals[np.ix_(i, j)].astype("float32"), units


def convert_units(var: str, field: np.ndarray, source: str, units: str = "") -> np.ndarray:
    if var in ("t2m", "d2m"):
        return field - 273.15
    if var == "tcc" and source == "ecmwf":  # ECMWF en fracción (0-1), GFS ya en %
        return field * 100.0
    if var == "tp":
        # IFS da metros; AIFS y GFS, kg/m² (= mm).
        return np.clip(field * 1000.0 if units.strip() == "m" else field, 0, None)
    return field
