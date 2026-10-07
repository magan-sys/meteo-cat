"""Rejillas regulares lat/lon sobre Catalunya y máscara del territorio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from shapely import contains_xy
from shapely.geometry import shape

from .config import REPO_ROOT

# Caja que contiene Catalunya con un pequeño margen.
LAT_MIN, LAT_MAX = 40.50, 42.90
LON_MIN, LON_MAX = 0.15, 3.35

BOUNDARY_PATH = REPO_ROOT / "data" / "catalunya.geojson"


@dataclass(frozen=True)
class Grid:
    step: float
    lat_min: float = LAT_MIN
    lat_max: float = LAT_MAX
    lon_min: float = LON_MIN
    lon_max: float = LON_MAX

    @cached_property
    def lats(self) -> np.ndarray:
        """Latitudes de los centros de celda, de norte a sur (orden de imagen)."""
        n = int(round((self.lat_max - self.lat_min) / self.step))
        return np.round(self.lat_max - self.step * (np.arange(n) + 0.5), 6)

    @cached_property
    def lons(self) -> np.ndarray:
        n = int(round((self.lon_max - self.lon_min) / self.step))
        return np.round(self.lon_min + self.step * (np.arange(n) + 0.5), 6)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lats), len(self.lons)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        lon2d, lat2d = np.meshgrid(self.lons, self.lats)
        return lat2d, lon2d

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max) del borde exterior de la rejilla."""
        ny, nx = self.shape
        return (self.lon_min, self.lat_max - ny * self.step, self.lon_min + nx * self.step, self.lat_max)


def catalunya_geometry():
    with open(BOUNDARY_PATH) as f:
        return shape(json.load(f)["features"][0]["geometry"])


def catalunya_mask(grid: Grid) -> np.ndarray:
    lat2d, lon2d = grid.mesh()
    return contains_xy(catalunya_geometry(), lon2d, lat2d)


def model_points(grid: Grid, buffer_deg: float = 0.15) -> np.ndarray:
    """Puntos (lat, lon) de la malla de modelos que caen en Catalunya o cerca de ella.

    Se incluye un margen para que la interpolación al borde de la rejilla fina tenga vecinos.
    """
    geom = catalunya_geometry().buffer(buffer_deg)
    lat2d, lon2d = grid.mesh()
    inside = contains_xy(geom, lon2d, lat2d)
    return np.column_stack([lat2d[inside], lon2d[inside]])
