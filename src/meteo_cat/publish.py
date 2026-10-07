"""Genera el panel como web estática (para GitHub Pages): HTML, imágenes por hora y series por punto.

Estructura:
  index.html
  data/meta.json                 metadatos, horas, rejilla, paleta
  data/png/t2m_<YYYYMMDDTHH>.png mapa de cada hora
  data/png/rain_<YYYYMMDDTHH>.png mapa de probabilidad de lluvia de cada hora
  data/series/<ti>_<tj>.bin      series por punto en bloques de TILE x TILE celdas:
                                 int16 little-endian [capa][horas][TILE][TILE], valores x10,
                                 capas en el orden de SERIES; -32768 = sin dato
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, Settings
from .grid import Grid
from .render import colorize_rain, palette_stops, rain_palette_stops, render_all

TILE = 32
MISSING = -32768
# Capas de las series por punto: (nombre, clave en latest.npz, factor antes de x10).
SERIES = (("t2m", "t2m", 1.0), ("t2m_raw", "t2m_raw", 1.0), ("rain_prob", "rain_prob", 100.0),
          ("rain_mm", "rain_mm", 1.0))


def to_int16(a: np.ndarray) -> np.ndarray:
    out = np.where(np.isfinite(a), np.round(a * 10), MISSING)
    return np.clip(out, MISSING, 32767).astype("<i2")


def publish(settings: Settings, out_dir: Path) -> Path:
    src = settings.forecast_dir
    meta = json.loads((src / "latest.json").read_text())
    d = np.load(src / "latest.npz")
    grid = Grid(settings.fine_grid_step)
    times = [np.datetime64(t[:19]) for t in meta["times"]]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    data = out_dir / "data"
    (data / "series").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "web" / "index.html", out_dir / "index.html")
    shutil.copy(REPO_ROOT / "data" / "catalunya.geojson", data / "catalunya.geojson")
    (out_dir / ".nojekyll").write_text("")

    import pandas as pd
    ts = [pd.Timestamp(t) for t in meta["times"]]
    meta["images"] = ["data/" + p for p in render_all(d["t2m"], ts, grid, data)]
    meta["palette"] = palette_stops()
    meta["rain_images"] = ["data/" + p for p in render_all(d["rain_prob"], ts, grid, data, "rain", colorize_rain)]
    meta["rain_palette"] = rain_palette_stops()

    layers = [to_int16(d[key] * f) for _, key, f in SERIES]
    ny, nx = grid.shape
    for ti in range(0, ny, TILE):
        for tj in range(0, nx, TILE):
            first = layers[0][:, ti:ti + TILE, tj:tj + TILE]
            if (first == MISSING).all():
                continue
            block = np.full((len(layers), len(times), TILE, TILE), MISSING, dtype="<i2")
            h, w = first.shape[1:]
            for k, arr in enumerate(layers):
                block[k, :, :h, :w] = arr[:, ti:ti + TILE, tj:tj + TILE]
            block.tofile(data / "series" / f"{ti // TILE}_{tj // TILE}.bin")

    meta["grid"] = {"lat0": float(grid.lats[0]), "lon0": float(grid.lons[0]), "step": grid.step,
                    "ny": ny, "nx": nx, "tile": TILE, "missing": MISSING,
                    "layers": [name for name, _, _ in SERIES]}
    (data / "meta.json").write_text(json.dumps(meta))
    return out_dir
