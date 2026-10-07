"""Flujo completo con datos sintéticos: entrenamiento, previsión con nowcasting y panel."""

import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from conftest import MODELS, FakeOpenMeteo, model_value, synthetic_elevation, truth
from meteo_cat.forecast import Observations, Pipeline
from meteo_cat.grid import Grid
from meteo_cat.history import station_terrain
from meteo_cat.sources.openmeteo import HISTORY_VARS
from meteo_cat.terrain import Terrain
from meteo_cat.train import fit


def synthetic_history(n_stations=40, days=60, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(40.8, 42.6, n_stations)
    lon = rng.uniform(0.5, 2.6, n_stations)
    stations = pd.DataFrame({"station": [f"S{i:02d}" for i in range(n_stations)], "lat": lat, "lon": lon,
                             "altitude": synthetic_elevation(lat, lon)})
    stations = stations[stations["altitude"] > 0].reset_index(drop=True)
    times = pd.date_range("2026-01-01", periods=days * 24, freq="h", tz="UTC")
    rows = []
    for lead in (0, 1, 2):
        for _, st in stations.iterrows():
            df = pd.DataFrame({"station": st.station, "time": times, "lead_day": lead,
                               "lat": st.lat, "lon": st.lon, "altitude": st.altitude})
            for m in MODELS:
                df[f"temperature_2m__{m}"] = model_value(m, np.full(len(times), st.lat), st.lon, times, lead, rng)
                df[f"cloud_cover__{m}"] = rng.uniform(0, 100, len(times))
                df[f"wind_speed_10m__{m}"] = rng.uniform(0, 30, len(times))
            df["obs"] = truth(np.full(len(times), st.lat), st.lon, times) + rng.normal(0, 0.3, len(times))
            rows.append(df)
    return stations, pd.concat(rows, ignore_index=True)


def test_train_forecast_and_serve(settings):
    stations, df = synthetic_history()
    terrain = Terrain(settings.terrain_dir / f"terrain_{Grid(settings.fine_grid_step).step:g}.npz")
    report = fit(df, station_terrain(df, terrain), settings.models, HISTORY_VARS, settings.models_dir)

    scores = {r["prediction"]: r["mae"] for r in report["estaciones_reservadas"]}
    # La corrección debe mejorar claramente a la media de modelos, también en estaciones no vistas.
    assert scores["corregido_ML"] < 0.5 * scores["media_modelos"]
    shift = report["desfase_horario_mae"]
    assert min(shift, key=shift.get) == 0
    assert (settings.models_dir / "t2m.txt").exists()

    # Observaciones "ahora" con un error común de +2 °C: el nowcasting debe absorberlo al inicio.
    now = pd.Timestamp("2026-10-07T06:00Z")
    obs = pd.DataFrame({"station": stations["station"], "time": now,
                        "value": truth(stations["lat"].to_numpy(), stations["lon"].to_numpy(), now) + 2.0})
    pipe = Pipeline(settings)
    meta = pipe.run(FakeOpenMeteo(), Observations(stations, obs), now=now)
    assert meta["ml_correction"] and meta["nowcast"]["stations"] == len(stations)
    assert len(meta["times"]) == 24 and meta["times"][0].startswith("2026-10-07T06")

    d = np.load(settings.forecast_dir / "latest.npz")
    t2m = d["t2m"]
    assert t2m.shape == (24,) + Grid(settings.fine_grid_step).shape
    lat2d, lon2d = Grid(settings.fine_grid_step).mesh()
    inside = np.isfinite(t2m[0])
    true0 = truth(lat2d[inside], lon2d[inside], now)
    true12 = truth(lat2d[inside], lon2d[inside], now + pd.Timedelta(hours=12))
    # Sin nowcasting la previsión corregida no tiene sesgo; el error de +2 °C de "ahora" se
    # incorpora cerca de las estaciones (no en todo el territorio) y desaparece a las 12 h.
    assert 0.4 < np.mean(t2m[0][inside] - true0) < 2.0
    assert abs(np.mean(t2m[12][inside] - true12)) < 0.3
    assert np.mean(np.abs(t2m[12][inside] - true12)) < 0.6
    assert np.mean(np.abs(d["t2m_raw"][12][inside] - true12)) > 1.0
    assert len(list((settings.forecast_dir / "png").glob("*.png"))) == 24

    client = TestClient(__import__("meteo_cat.server", fromlist=["create_app"]).create_app())
    m = client.get("/api/meta").json()
    assert len(m["images"]) == 24 and client.get(m["images"][0]).status_code == 200
    p = client.get("/api/point", params={"lat": 41.4, "lon": 2.1}).json()
    assert len(p["corregido"]) == 24 and p["corregido"][0] is not None
    assert client.get("/api/point", params={"lat": 40.6, "lon": 3.3}).status_code == 404
    assert "Leaflet" in client.get("/").text or "leaflet" in client.get("/").text
    json.dumps(report, default=str)
