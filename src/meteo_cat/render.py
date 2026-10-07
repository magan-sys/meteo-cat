"""Imágenes PNG de la temperatura por hora para el panel web.

Escala divergente azul-gris-rojo centrada en 15 °C (frío / cálido), de -10 a 40 °C.
El brazo rojo se genera con la misma luminosidad perceptual (OKLab) que el azul, para que
ambos lados pesen igual visualmente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VMIN, VMID, VMAX = -10.0, 15.0, 40.0
BLUE_ARM = ["#0d366b", "#256abf", "#6da7ec", "#b7d3f6"]  # de extremo a centro
NEUTRAL = "#f0efec"
RED_HUE_REF = "#e34948"


def _hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def _rgb_to_hex(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(round(np.clip(c, 0, 1) * 255)):02x}" for c in rgb)


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _delin(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def to_oklab(rgb: np.ndarray) -> np.ndarray:
    return _M2 @ np.cbrt(_M1 @ _lin(rgb))


def from_oklab(lab: np.ndarray) -> np.ndarray:
    lms = np.linalg.solve(_M2, lab) ** 3
    return _delin(np.linalg.solve(_M1, lms))


def _red_arm() -> list[str]:
    """Rojos con la misma L y croma que cada paso azul, con el tono del rojo de referencia."""
    ref = to_oklab(_hex_to_rgb(RED_HUE_REF))
    hue = np.arctan2(ref[2], ref[1])
    out = []
    for h in BLUE_ARM:
        L, a, b = to_oklab(_hex_to_rgb(h))
        C = np.hypot(a, b)
        for _ in range(40):  # reduce croma hasta que el color quepa en sRGB
            rgb = from_oklab(np.array([L, C * np.cos(hue), C * np.sin(hue)]))
            if np.all((rgb >= -1e-4) & (rgb <= 1 + 1e-4)):
                break
            C *= 0.95
        out.append(_rgb_to_hex(rgb))
    return out


def palette_stops() -> list[tuple[float, str]]:
    """(temperatura, color) de -10 a 40 °C."""
    colors = BLUE_ARM + [NEUTRAL] + list(reversed(_red_arm()))
    temps = np.linspace(VMIN, VMAX, len(colors))
    return [(float(t), c) for t, c in zip(temps, colors)]


def colorize(values: np.ndarray) -> np.ndarray:
    """Array de temperaturas -> RGBA uint8 (transparente donde hay NaN)."""
    stops = palette_stops()
    t = np.array([s[0] for s in stops])
    labs = np.array([to_oklab(_hex_to_rgb(c)) for _, c in stops])
    v = np.clip(values, VMIN, VMAX)
    lab = np.stack([np.interp(v, t, labs[:, k]) for k in range(3)], axis=-1)
    flat = lab.reshape(-1, 3).T
    rgb = _delin(np.linalg.solve(_M1, np.linalg.solve(_M2, flat) ** 3)).T.reshape(values.shape + (3,))
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = np.round(np.clip(np.nan_to_num(rgb), 0, 1) * 255)
    rgba[..., 3] = np.where(np.isfinite(values), 255, 0)
    return rgba


def to_mercator_rows(field: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Remuestrea las filas para que la imagen quede bien alineada en un mapa web (Mercator)."""
    def merc(lat):
        return np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))
    step = abs(lats[1] - lats[0])
    top, bottom = lats[0] + step / 2, lats[-1] - step / 2
    y = np.linspace(merc(top), merc(bottom), len(lats) + 1)
    y = (y[:-1] + y[1:]) / 2
    lat_rows = np.rad2deg(2 * np.arctan(np.exp(y)) - np.pi / 2)
    idx = np.clip(np.round((lats[0] - lat_rows) / step).astype(int), 0, len(lats) - 1)
    return field[idx]


def render_all(fields: np.ndarray, times, grid, out_dir: Path) -> None:
    import matplotlib.image as mpimg

    img_dir = out_dir / "png"
    img_dir.mkdir(parents=True, exist_ok=True)
    for k, t in enumerate(times):
        rgba = colorize(to_mercator_rows(fields[k], grid.lats))
        mpimg.imsave(img_dir / f"t2m_{t.strftime('%Y%m%dT%H')}.png", rgba)
