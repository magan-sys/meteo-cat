"""Servidor del panel web: ficheros estáticos + API de consulta por punto."""

from __future__ import annotations

import json

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .config import REPO_ROOT, get_settings
from .render import palette_stops


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="meteo-cat")
    fdir = settings.forecast_dir
    cache: dict = {}

    def load():
        npz_path = fdir / "latest.npz"
        if not npz_path.exists():
            raise HTTPException(404, "Todavía no hay ninguna previsión. Ejecuta `meteo-cat forecast`.")
        mtime = npz_path.stat().st_mtime
        if cache.get("mtime") != mtime:
            d = np.load(npz_path)
            cache.update(mtime=mtime, lats=d["lats"], lons=d["lons"], t2m=d["t2m"], raw=d["t2m_raw"],
                         times=[str(t) for t in d["times"]])
        return cache

    @app.get("/api/meta")
    def meta():
        path = fdir / "latest.json"
        if not path.exists():
            raise HTTPException(404, "Todavía no hay ninguna previsión. Ejecuta `meteo-cat forecast`.")
        m = json.loads(path.read_text())
        m["images"] = [f"/forecast/png/t2m_{t[:13].replace('-', '').replace(':', '')}.png" for t in m["times"]]
        m["palette"] = palette_stops()
        return m

    @app.get("/api/point")
    def point(lat: float, lon: float):
        c = load()
        step = abs(float(c["lons"][1] - c["lons"][0]))
        i = int(round((c["lats"][0] - lat) / step))
        j = int(round((lon - c["lons"][0]) / step))
        if not (0 <= i < len(c["lats"]) and 0 <= j < len(c["lons"])):
            raise HTTPException(404, "Fuera de la zona de previsión")
        corr = c["t2m"][:, i, j]
        if not np.isfinite(corr).any():
            raise HTTPException(404, "Fuera de Catalunya")

        def clean(a):
            return [None if not np.isfinite(v) else round(float(v), 1) for v in a]

        return {"lat": float(c["lats"][i]), "lon": float(c["lons"][j]), "times": c["times"],
                "corregido": clean(corr), "media_modelos": clean(c["raw"][:, i, j])}

    fdir.mkdir(parents=True, exist_ok=True)
    app.mount("/forecast", StaticFiles(directory=fdir), name="forecast")
    app.mount("/", StaticFiles(directory=REPO_ROOT / "web", html=True), name="web")
    return app
