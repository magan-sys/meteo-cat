"""Interpolación de puntos dispersos a la rejilla fina con pesos precalculados."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree


class LinearInterpolator:
    """Interpolación lineal (triangulación de Delaunay); fuera de la envolvente, vecino más cercano.

    Los pesos se calculan una vez y se reutilizan para todas las variables y horas.
    """

    def __init__(self, src: np.ndarray, dst: np.ndarray):
        tri = Delaunay(src)
        simplex = tri.find_simplex(dst)
        inside = simplex >= 0
        self.idx = np.zeros((len(dst), 3), dtype=np.int64)
        self.w = np.zeros((len(dst), 3), dtype=np.float64)
        T = tri.transform[simplex[inside]]
        b = np.einsum("nij,nj->ni", T[:, :2, :], dst[inside] - T[:, 2, :])
        self.idx[inside] = tri.simplices[simplex[inside]]
        self.w[inside] = np.column_stack([b, 1 - b.sum(axis=1)])
        if (~inside).any():
            _, nn = cKDTree(src).query(dst[~inside])
            self.idx[~inside, 0] = nn
            self.w[~inside, 0] = 1.0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """values: (n_src,) o (n_src, k) -> (n_dst,) o (n_dst, k). NaN si algún vértice es NaN."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 1:
            return np.einsum("nk,nk->n", v[self.idx], self.w)
        return np.einsum("nkf,nk->nf", v[self.idx], self.w)


def idw_residuals(st_lat, st_lon, st_elev, resid, lat, lon, elev,
                  length_km: float = 25.0, elev_scale_m: float = 50.0, power: float = 2.0) -> np.ndarray:
    """Interpola residuos de estaciones con distancia inversa que penaliza diferencias de altitud.

    50 m de desnivel cuentan como 1 km de distancia horizontal, de modo que un valle no toma
    el error de una cumbre cercana. Más allá de `length_km` el peso decae a cero.
    """
    st_lat, st_lon, st_elev, resid = map(np.asarray, (st_lat, st_lon, st_elev, resid))
    ok = np.isfinite(resid)
    st_lat, st_lon, st_elev, resid = st_lat[ok], st_lon[ok], st_elev[ok], resid[ok]
    if len(resid) == 0:
        return np.zeros(len(lat))
    kx = 111.32 * np.cos(np.deg2rad(np.mean(st_lat)))
    src = np.column_stack([st_lat * 111.32, st_lon * kx, st_elev / elev_scale_m])
    dst = np.column_stack([np.asarray(lat) * 111.32, np.asarray(lon) * kx, np.asarray(elev) / elev_scale_m])
    k = min(8, len(resid))
    d, i = cKDTree(src).query(dst, k=k)
    d, i = d.reshape(len(dst), k), i.reshape(len(dst), k)
    w = 1.0 / np.maximum(d, 0.5) ** power
    w *= np.clip(1.0 - d / length_km, 0.0, 1.0)  # decaimiento con la distancia
    wsum = w.sum(axis=1)
    out = np.where(wsum > 0, (w * resid[i]).sum(axis=1) / np.where(wsum > 0, wsum, 1.0), 0.0)
    # Lejos de cualquier estación el residuo tiende a 0 (se confía en la previsión corregida).
    support = np.clip(d.min(axis=1) / length_km, 0.0, 1.0)
    return out * (1.0 - support)
