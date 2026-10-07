"""Variables de terreno a partir del DEM Copernicus GLO-90 (90 m, acceso público en AWS).

Genera, sobre la rejilla fina (~1 km):
  elevation       altitud media de la celda (m)
  elev_std        rugosidad: desviación típica de la altitud dentro de la celda (m)
  slope           pendiente (grados)
  aspect_sin/cos  orientación de la ladera
  tpi_5km         posición topográfica: altitud menos la media a ~5 km (valles < 0, crestas > 0)
  elev_smooth_10km altitud suavizada a ~10 km
  elev_smooth_25km altitud suavizada a la escala de los modelos globales (0,25°)
  land_frac       fracción de tierra
  dist_sea_km     distancia al mar (km)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import requests
from scipy import ndimage

from .grid import Grid

log = logging.getLogger(__name__)

DEM_URL = (
    "https://copernicus-dem-90m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)

TERRAIN_FEATURES = (
    "elevation",
    "elev_std",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_5km",
    "elev_smooth_10km",
    "elev_smooth_25km",
    "land_frac",
    "dist_sea_km",
)

KM_PER_DEG_LAT = 111.32


def _tile_url(lat: int, lon: int) -> str:
    return DEM_URL.format(
        ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon)
    )


def download_dem_tiles(grid: Grid, dest: Path) -> list[Path]:
    """Descarga las teselas 1x1 grado que cubren la rejilla. Las teselas solo de mar no existen (404)."""
    dest.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = grid.bounds
    paths = []
    for lat in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            url = _tile_url(lat, lon)
            path = dest / Path(url).name
            if not path.exists():
                r = requests.get(url, timeout=120)
                if r.status_code in (403, 404):
                    log.info("Tesela sin datos (mar): %s", path.name)
                    continue
                r.raise_for_status()
                path.write_bytes(r.content)
                log.info("Descargada %s", path.name)
            paths.append(path)
    return paths


def _resample_to_grid(tile_paths: list[Path], grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media, media de cuadrados y fracción de tierra de los píxeles de 90 m en cada celda."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    ny, nx = grid.shape
    dst_transform = from_origin(grid.lon_min, grid.lat_max, grid.step, grid.step)
    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, src_transform = merge(sources, nodata=0)
        crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
    dem = mosaic[0].astype("float64")
    dem = np.where(np.isfinite(dem), dem, 0.0)

    out = {}
    for name, arr in {"mean": dem, "sq": dem**2, "land": (dem > 0).astype("float64")}.items():
        dst = np.zeros((ny, nx), dtype="float64")
        reproject(
            arr, dst,
            src_transform=src_transform, src_crs=crs,
            dst_transform=dst_transform, dst_crs=crs,
            resampling=Resampling.average,
        )
        out[name] = dst
    return out["mean"], out["sq"], out["land"]


def derive_features(elev: np.ndarray, elev_sq: np.ndarray, land_frac: np.ndarray, grid: Grid) -> dict[str, np.ndarray]:
    """Calcula las variables de terreno a partir de la altitud media por celda."""
    lat_c = np.deg2rad(np.mean(grid.lats))
    dy = grid.step * KM_PER_DEG_LAT * 1000.0  # m
    dx = grid.step * KM_PER_DEG_LAT * np.cos(lat_c) * 1000.0  # m

    # Las filas van de norte a sur: gradiente hacia el norte = -d/dfila.
    dz_drow, dz_dcol = np.gradient(elev)
    dz_dn = -dz_drow / dy
    dz_de = dz_dcol / dx
    slope = np.degrees(np.arctan(np.hypot(dz_dn, dz_de)))
    # Orientación: dirección hacia la que baja la ladera (0 = norte, 90 = este).
    aspect = np.arctan2(-dz_de, -dz_dn)
    flat = slope < 0.5
    aspect_sin = np.where(flat, 0.0, np.sin(aspect))
    aspect_cos = np.where(flat, 0.0, np.cos(aspect))

    cell_km = grid.step * KM_PER_DEG_LAT
    smooth5 = ndimage.gaussian_filter(elev, sigma=5.0 / cell_km, mode="nearest")
    smooth10 = ndimage.gaussian_filter(elev, sigma=10.0 / cell_km, mode="nearest")
    # Una celda de 0,25° (~25 km) equivale aproximadamente a un filtro gaussiano de sigma ~12 km.
    smooth25 = ndimage.gaussian_filter(elev, sigma=12.0 / cell_km, mode="nearest")

    sea = land_frac < 0.5
    if sea.any():
        dist = ndimage.distance_transform_edt(~sea, sampling=(dy / 1000.0, dx / 1000.0))
    else:
        dist = np.full(elev.shape, 999.0)

    return {
        "elevation": elev,
        "elev_std": np.sqrt(np.clip(elev_sq - elev**2, 0, None)),
        "slope": slope,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
        "tpi_5km": elev - smooth5,
        "elev_smooth_10km": smooth10,
        "elev_smooth_25km": smooth25,
        "land_frac": land_frac,
        "dist_sea_km": dist,
    }


def build_terrain(grid: Grid, terrain_dir: Path) -> Path:
    tiles = download_dem_tiles(grid, terrain_dir / "dem")
    if not tiles:
        raise RuntimeError("No se ha podido descargar ninguna tesela del DEM")
    elev, elev_sq, land = _resample_to_grid(tiles, grid)
    feats = derive_features(elev, elev_sq, land, grid)
    out = terrain_dir / f"terrain_{grid.step:g}.npz"
    np.savez_compressed(out, lats=grid.lats, lons=grid.lons, **{k: v.astype("float32") for k, v in feats.items()})
    log.info("Terreno guardado en %s", out)
    return out


class Terrain:
    """Variables de terreno en memoria, con consulta por coordenadas."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.lats = d["lats"]
        self.lons = d["lons"]
        self.fields = {k: d[k] for k in TERRAIN_FEATURES}

    def index(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        step = abs(self.lons[1] - self.lons[0])
        i = np.clip(np.round((self.lats[0] - np.asarray(lat)) / step).astype(int), 0, len(self.lats) - 1)
        j = np.clip(np.round((np.asarray(lon) - self.lons[0]) / step).astype(int), 0, len(self.lons) - 1)
        return i, j

    def at(self, lat, lon) -> dict[str, np.ndarray]:
        i, j = self.index(lat, lon)
        return {k: v[i, j] for k, v in self.fields.items()}
