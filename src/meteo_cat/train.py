"""Entrenamiento del corrector de temperatura (LightGBM) y su verificación."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import TARGET_VAR, build_features
from .verify import compare, persistence_tau, time_alignment

log = logging.getLogger(__name__)

PARAMS = {
    "objective": "huber",
    "alpha": 2.0,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
MAX_ROUNDS = 3000


def split(df: pd.DataFrame, holdout_frac: float = 0.2, time_frac: float = 0.8, seed: int = 0):
    """Partición temporal y espacial.

    - train: primer 80 % del periodo, estaciones no reservadas
    - val:   último 20 % del periodo, mismas estaciones (parada temprana y métrica temporal)
    - test:  último 20 % del periodo, estaciones reservadas (simula puntos sin estación)
    """
    times = np.sort(df["time"].unique())
    cutoff = times[int(len(times) * time_frac)]
    stations = np.array(sorted(df["station"].unique()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(stations, size=max(1, int(len(stations) * holdout_frac)), replace=False))
    is_held = df["station"].isin(held).to_numpy()
    late = (df["time"] >= cutoff).to_numpy()
    return (~late & ~is_held), (late & ~is_held), (late & is_held)


def fit(df: pd.DataFrame, terrain: dict[str, np.ndarray], models: tuple[str, ...], out_dir: Path) -> dict:
    X, base = build_features(df, terrain, models)
    y = df["obs"].to_numpy(dtype="float64") - base
    ok = np.isfinite(y)
    df, X, base, y = df[ok].reset_index(drop=True), X[ok].reset_index(drop=True), base[ok], y[ok]
    if len(df) < 1000:
        raise RuntimeError(f"Muy pocos datos para entrenar ({len(df)} filas)")

    tr, va, te = split(df)
    dtr = lgb.Dataset(X[tr], y[tr])
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    booster = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    best = booster.best_iteration or MAX_ROUNDS

    eval_df = df[["station", "time", "lead_h", "obs"]].copy()
    eval_df["lead_day"] = (eval_df["lead_h"] // 24).clip(upper=2).astype(int)
    eval_df["raw_ens_mean"] = X["ens_mean"].to_numpy()
    eval_df["base"] = base
    eval_df["corrected"] = base + booster.predict(X, num_iteration=best)
    preds = {"media_modelos": "raw_ens_mean", "media+altitud": "base", "corregido_ML": "corrected"}
    for m in models:
        col = f"{TARGET_VAR}__{m}"
        if col in df:
            eval_df[m] = df[col].to_numpy()
            preds[m] = m

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": [str(df["time"].min()), str(df["time"].max())],
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test_estaciones_reservadas": int(te.sum())},
        "best_iteration": int(best),
        "validacion_temporal": compare(eval_df[va], "obs", preds).to_dict("records"),
        "validacion_por_horizonte": compare(eval_df[va], "obs", preds, by="lead_day").to_dict("records"),
        "estaciones_reservadas": compare(eval_df[te], "obs", preds).to_dict("records"),
        "desfase_horario_mae": time_alignment(eval_df[eval_df["lead_day"] == 0], "obs", "raw_ens_mean"),
    }
    eval_df["resid"] = eval_df["obs"] - eval_df["corrected"]
    # Para la persistencia del error se usa la pasada más reciente posible (primeras 24 h).
    tau = persistence_tau(eval_df[va & (eval_df["lead_day"] == 0).to_numpy()], "resid")
    report["nowcast_tau_h"] = tau

    # Modelo final con todos los datos y el número de iteraciones hallado.
    final = lgb.train(PARAMS, lgb.Dataset(X, y), int(best * 1.1))
    out_dir.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out_dir / "t2m.txt"))
    meta = {"target": TARGET_VAR, "models": list(models), "features": list(X.columns),
            "nowcast_tau_h": tau, "trained_at": report["trained_at"], "period": report["period"]}
    (out_dir / "t2m_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "t2m_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


class Corrector:
    """Modelo entrenado cargado de disco. Sin modelo, devuelve la base (media + altitud)."""

    def __init__(self, models_dir: Path):
        meta_path = models_dir / "t2m_meta.json"
        self.booster = None
        self.meta = None
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
            self.booster = lgb.Booster(model_file=str(models_dir / "t2m.txt"))

    @property
    def tau_h(self) -> float:
        return float(self.meta["nowcast_tau_h"]) if self.meta else 3.0

    def predict(self, X: pd.DataFrame, base: np.ndarray) -> np.ndarray:
        if self.booster is None:
            return base
        return base + self.booster.predict(X[self.meta["features"]])


def format_report(report: dict) -> str:
    lines = [f"Periodo: {report['period'][0]} a {report['period'][1]}", ""]
    for key, title in (("validacion_temporal", "Validación (últimos meses, estaciones de entrenamiento)"),
                       ("estaciones_reservadas", "Estaciones no vistas en el entrenamiento")):
        lines += [title, f"{'predicción':<32}{'MAE':>7}{'RMSE':>7}{'sesgo':>7}"]
        for r in sorted(report[key], key=lambda r: r["mae"]):
            lines.append(f"{r['prediction']:<32}{r['mae']:7.2f}{r['rmse']:7.2f}{r['bias']:7.2f}")
        lines.append("")
    shift = report["desfase_horario_mae"]
    best = int(min(shift, key=lambda k: shift[k]))
    if best != 0:
        lines.append(f"AVISO: las observaciones encajan mejor con un desfase de {best} h. Revisar zona horaria.")
    lines.append(f"Persistencia del error para nowcasting: tau = {report['nowcast_tau_h']:.1f} h")
    return "\n".join(lines)
