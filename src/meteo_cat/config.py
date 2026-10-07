"""Configuración leída de variables de entorno. Ninguna clave va en el código."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Modelos numéricos consultados en Open-Meteo. Cada uno cubre un horizonte distinto;
# los valores ausentes (p. ej. AROME más allá de ~42 h) llegan como NaN y LightGBM los admite.
DEFAULT_MODELS = (
    "meteofrance_arome_france_hd",  # ~1,5 km, hasta ~42 h
    "meteofrance_arpege_europe",    # ~11 km, hasta ~4 días
    "icon_d2",                      # ~2 km, hasta 48 h
    "icon_eu",                      # ~7 km, hasta 5 días
    "ecmwf_ifs025",                 # ~25 km, hasta 15 días
)


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("METEOCAT_DATA_DIR", REPO_ROOT / "var")))
    # Open-Meteo: el uso gratuito es solo no comercial. Con clave se usan los servidores "customer-".
    open_meteo_api_key: str | None = field(default_factory=lambda: os.environ.get("OPEN_METEO_API_KEY") or None)
    # Reservadas para la siguiente fase (API XEMA de Meteocat y AEMET OpenData).
    meteocat_api_key: str | None = field(default_factory=lambda: os.environ.get("METEOCAT_API_KEY") or None)
    aemet_api_key: str | None = field(default_factory=lambda: os.environ.get("AEMET_API_KEY") or None)
    models: tuple[str, ...] = field(default_factory=lambda: _env_list("METEOCAT_MODELS", DEFAULT_MODELS))
    # Resolución (grados) de la malla donde se piden los modelos en operación.
    model_grid_step: float = field(default_factory=lambda: float(os.environ.get("METEOCAT_MODEL_GRID_STEP", "0.1")))
    # Resolución (grados) de la rejilla final (~1 km).
    fine_grid_step: float = field(default_factory=lambda: float(os.environ.get("METEOCAT_FINE_GRID_STEP", "0.01")))
    forecast_hours: int = field(default_factory=lambda: int(os.environ.get("METEOCAT_FORECAST_HOURS", "72")))

    @property
    def terrain_dir(self) -> Path:
        return self.data_dir / "terrain"

    @property
    def history_dir(self) -> Path:
        return self.data_dir / "history"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def forecast_dir(self) -> Path:
        return self.data_dir / "forecast"


def get_settings() -> Settings:
    return Settings()
