"""Genera el panel como web estática (para GitHub Pages): HTML, imágenes por hora y series por punto.

Estructura:
  index.html
  data/meta.json                 metadatos, horas, rejilla, paleta
  data/png/t2m_<YYYYMMDDTHH>.png mapa de cada hora
  data/series/<ti>_<tj>.bin      series por punto en bloques de TILE x TILE celdas:
                                 int16 little-endian [2][horas][TILE][TILE] en décimas de °C
                                 (0 = corregida, 1 = media de modelos), -32768 = sin dato
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, Settings
from .grid import Grid
from .render import palette_stops, render_all

TILE = 32
MISSING = -32768


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
    (out_dir / ".nojekyll").write_text("")

    import pandas as pd
    ts = [pd.Timestamp(t) for t in meta["times"]]
    render_all(d["t2m"], ts, grid, data)
    meta["images"] = [f"data/png/t2m_{t.strftime('%Y%m%dT%H')}.png" for t in ts]
    meta["palette"] = palette_stops()

    corr, raw = to_int16(d["t2m"]), to_int16(d["t2m_raw"])
    ny, nx = grid.shape
    for ti in range(0, ny, TILE):
        for tj in range(0, nx, TILE):
            block = np.full((2, len(times), TILE, TILE), MISSING, dtype="<i2")
            c = corr[:, ti:ti + TILE, tj:tj + TILE]
            if (c == MISSING).all():
                continue
            h, w = c.shape[1:]
            block[0, :, :h, :w] = c
            block[1, :, :h, :w] = raw[:, ti:ti + TILE, tj:tj + TILE]
            block.tofile(data / "series" / f"{ti // TILE}_{tj // TILE}.bin")

    meta["grid"] = {"lat0": float(grid.lats[0]), "lon0": float(grid.lons[0]), "step": grid.step,
                    "ny": ny, "nx": nx, "tile": TILE, "missing": MISSING}
    (data / "meta.json").write_text(json.dumps(meta))
    return out_dir
