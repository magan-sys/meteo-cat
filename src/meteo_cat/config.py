"""Configuración leída de variables de entorno. Ninguna clave va en el código."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Modelos numéricos abiertos (ver sources/nwp.py). AROME e ICON-EU no tienen histórico
# público accesible y quedan para una fase posterior.
DEFAULT_MODELS = ("ecmwf_ifs", "ecmwf_aifs", "gfs")


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("METEOCAT_DATA_DIR", REPO_ROOT / "var")))
    models: tuple[str, ...] = field(default_factory=lambda: _env_list("METEOCAT_MODELS", DEFAULT_MODELS))
    # Resolución (grados) de la rejilla final (~1 km).
    fine_grid_step: float = field(default_factory=lambda: float(os.environ.get("METEOCAT_FINE_GRID_STEP", "0.01")))
    forecast_hours: int = field(default_factory=lambda: int(os.environ.get("METEOCAT_FORECAST_HOURS", "72")))

    @property
    def terrain_dir(self) -> Path:
        return self.data_dir / "terrain"

    @property
    def history_dir(self) -> Path:
        return self.data_dir / "history"

    # El modelo entrenado se guarda en el repositorio para que lo use el servicio programado.
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("METEOCAT_MODELS_DIR", REPO_ROOT / "models")))

    @property
    def forecast_dir(self) -> Path:
        return self.data_dir / "forecast"


def get_settings() -> Settings:
    return Settings()
