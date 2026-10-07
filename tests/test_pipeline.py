"""Flujo completo con datos sintéticos: entrenamiento, previsión con nowcasting y web estática."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from conftest import MODELS, FakeNWP, model_value, synthetic_elevation, truth
from meteo_cat.forecast import Observations, Pipeline
from meteo_cat.grid import Grid
from meteo_cat.history import station_terrain
from meteo_cat.publish import MISSING, TILE, publish
from meteo_cat.terrain import Terrain
from meteo_cat.train import fit


def synthetic_history(n_stations=40, days=60, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(40.8, 42.6, n_stations)
    lon = rng.uniform(0.5, 2.6, n_stations)
    stations = pd.DataFrame({"station": [f"S{i:02d}" for i in range(n_stations)], "lat": lat, "lon": lon,
                             "altitude": synthetic_elevation(lat, lon)})
    stations = stations[stations["altitude"] > 0].reset_index(drop=True)
    runs = pd.date_range("2026-01-01", periods=days, freq="D", tz="UTC")
    rows = []
    for lead in range(0, 73, 3):
        times = runs + pd.Timedelta(hours=lead)
        for _, st in stations.iterrows():
            df = pd.DataFrame({"station": st.station, "time": times, "lead_h": lead,
                               "lat": st.lat, "lon": st.lon, "altitude": st.altitude})
            for m in MODELS:
                t = model_value(m, np.full(len(times), st.lat), st.lon, times, lead, rng)
                df[f"t2m__{m}"] = t
                df[f"d2m__{m}"] = t - 5
                df[f"tcc__{m}"] = rng.uniform(0, 100, len(times))
                df[f"u10__{m}"] = rng.normal(0, 3, len(times))
                df[f"v10__{m}"] = rng.normal(0, 3, len(times))
            df["obs"] = truth(np.full(len(times), st.lat), st.lon, times) + rng.normal(0, 0.3, len(times))
            rows.append(df)
    return stations, pd.concat(rows, ignore_index=True)


def test_train_forecast_and_publish(settings, tmp_path):
    stations, df = synthetic_history()
    terrain = Terrain(settings.terrain_dir / f"terrain_{Grid(settings.fine_grid_step).step:g}.npz")
    report = fit(df, station_terrain(df, terrain), settings.models, settings.models_dir)

    scores = {r["prediction"]: r["mae"] for r in report["estaciones_reservadas"]}
    # La corrección debe mejorar claramente a la media de modelos, también en estaciones no vistas.
    assert scores["corregido_ML"] < 0.5 * scores["media_modelos"]
    shift = report["desfase_horario_mae"]
    assert min(shift, key=shift.get) == 0
    assert (settings.models_dir / "t2m.txt").exists()
    json.dumps(report, default=str)

    # Observaciones "ahora" con un error común de +2 °C: el nowcasting debe absorberlo al inicio.
    now = pd.Timestamp("2026-10-07T06:00Z")
    obs = pd.DataFrame({"station": stations["station"], "time": now,
                        "value": truth(stations["lat"].to_numpy(), stations["lon"].to_numpy(), now) + 2.0})
    client = FakeNWP(datetime(2026, 10, 7, 0, tzinfo=timezone.utc))
    meta = Pipeline(settings).run(client, Observations(stations, obs), now=now)
    assert meta["ml_correction"] and meta["nowcast"]["stations"] == len(stations)
    assert set(meta["runs"]) == set(MODELS)
    assert len(meta["times"]) == 24 and meta["times"][0].startswith("2026-10-07T06")

    d = np.load(settings.forecast_dir / "latest.npz")
    t2m = d["t2m"]
    grid = Grid(settings.fine_grid_step)
    assert t2m.shape == (24,) + grid.shape
    lat2d, lon2d = grid.mesh()
    inside = np.isfinite(t2m[0])
    true0 = truth(lat2d[inside], lon2d[inside], now)
    true12 = truth(lat2d[inside], lon2d[inside], now + pd.Timedelta(hours=12))
    # El error de +2 °C de "ahora" se incorpora cerca de las estaciones y desaparece a las 12 h.
    assert 0.4 < np.mean(t2m[0][inside] - true0) < 2.0
    assert abs(np.mean(t2m[12][inside] - true12)) < 0.3
    assert np.mean(np.abs(t2m[12][inside] - true12)) < 0.6
    assert np.mean(np.abs(d["t2m_raw"][12][inside] - true12)) > 1.0

    # Web estática: imágenes por hora y series por punto coherentes con la rejilla.
    site = publish(settings, tmp_path / "site")
    m = json.loads((site / "data" / "meta.json").read_text())
    assert len(m["images"]) == 24 and all((site / p).exists() for p in m["images"])
    assert (site / "index.html").exists() and "data/meta.json" in (site / "index.html").read_text()
    i, j = int(np.argwhere(inside)[0][0]), int(np.argwhere(inside)[0][1])
    tile = np.fromfile(site / "data" / "series" / f"{i // TILE}_{j // TILE}.bin", dtype="<i2")
    tile = tile.reshape(2, 24, TILE, TILE)
    assert tile[0, 5, i % TILE, j % TILE] == round(float(t2m[5, i, j]) * 10)
    assert tile[0, 0, 0, 0] == MISSING or np.isfinite(t2m[0, (i // TILE) * TILE, (j // TILE) * TILE])
