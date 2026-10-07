import numpy as np

from meteo_cat.grid import Grid, catalunya_mask
from meteo_cat.interp import LinearInterpolator, idw_residuals
from meteo_cat.render import colorize, palette_stops
from meteo_cat.sources.openmeteo import OpenMeteoClient, parse_hourly, parse_previous_runs
from meteo_cat.terrain import derive_features


def test_grid_and_mask():
    g = Grid(0.05)
    assert g.shape == (48, 64)
    m = catalunya_mask(g)
    lat2d, lon2d = g.mesh()
    # Barcelona dentro, Mallorca y Zaragoza fuera.
    def at(lat, lon):
        return m[np.argmin(abs(g.lats - lat)), np.argmin(abs(g.lons - lon))]
    assert at(41.40, 2.15) and at(41.62, 0.62)
    assert not at(41.0, 3.2)


def test_parse_hourly_multi_model():
    hourly = {"time": ["2026-10-07T00:00", "2026-10-07T01:00"],
              "temperature_2m_icon_eu": [10.0, 11.0], "temperature_2m_icon_d2": [9.0, None]}
    df = parse_hourly(hourly, ("temperature_2m",), ("icon_eu", "icon_d2"), point=3)
    assert set(df["model"]) == {"icon_eu", "icon_d2"}
    d2 = df[df["model"] == "icon_d2"]
    assert d2["temperature_2m"].iloc[0] == 9.0 and np.isnan(d2["temperature_2m"].iloc[1])
    assert (df["point"] == 3).all()
    assert str(df["time"].dt.tz) == "UTC"


def test_parse_previous_runs():
    hourly = {"time": ["2026-10-07T00:00"],
              "temperature_2m_icon_eu": [10.0],
              "temperature_2m_previous_day1_icon_eu": [11.0],
              "temperature_2m_previous_day2_icon_eu": [12.0],
              "cloud_cover_previous_day1_icon_eu": [50.0]}
    df = parse_previous_runs(hourly, ("temperature_2m", "cloud_cover"), ("icon_eu",), (0, 1, 2), point=0)
    got = df.set_index("lead_day")["temperature_2m"].to_dict()
    assert got == {0: 10.0, 1: 11.0, 2: 12.0}
    assert df.set_index("lead_day").loc[1, "cloud_cover"] == 50.0


def test_client_chunks_points_and_sends_elevation_nan():
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, n):
            self.n = n

        def json(self):
            return [{"hourly": {"time": ["2026-10-07T00:00"], "temperature_2m_a": [1.0],
                                "temperature_2m_b": [2.0]}} for _ in range(self.n)]

    class Session:
        def get(self, url, params, timeout):
            calls.append(params)
            return Resp(len(params["latitude"].split(",")))

    pts = np.column_stack([np.linspace(41, 42, 250), np.linspace(1, 2, 250)])
    df = OpenMeteoClient(session=Session(), pause_s=0).forecast(pts, ("a", "b"), 24, ("temperature_2m",))
    assert len(calls) == 3
    assert all(set(c["elevation"].split(",")) == {"nan"} for c in calls)
    assert df["point"].nunique() == 250 and len(df) == 500


def test_terrain_slope_and_aspect():
    g = Grid(0.01)
    lat2d, lon2d = g.mesh()
    # Plano que sube hacia el norte: la ladera mira al sur (aspect ~ 180°).
    elev = (lat2d - g.lat_min) * 1000.0
    f = derive_features(elev, elev**2, np.ones_like(elev), g)
    inner = (slice(5, -5), slice(5, -5))
    expected = np.degrees(np.arctan(1000.0 / 111320.0))
    assert np.allclose(f["slope"][inner], expected, atol=0.01)
    assert np.allclose(f["aspect_cos"][inner], -1.0, atol=1e-6)


def test_distance_to_sea():
    g = Grid(0.05)
    land = np.ones(g.shape)
    land[:, -1] = 0  # mar en la columna más al este
    f = derive_features(np.zeros(g.shape), np.zeros(g.shape), land, g)
    d = f["dist_sea_km"][10]
    assert d[-1] == 0 and d[-2] > 3 and np.all(np.diff(d[:-1]) < 0)


def test_linear_interpolator_exact_for_planes():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1, (200, 2))
    dst = rng.uniform(0.2, 0.8, (500, 2))
    f = lambda p: 3 * p[:, 0] - 2 * p[:, 1] + 1
    out = LinearInterpolator(src, dst)(np.column_stack([f(src), 2 * f(src)]))
    assert np.allclose(out[:, 0], f(dst)) and np.allclose(out[:, 1], 2 * f(dst))


def test_idw_residuals_local_and_decaying():
    r = idw_residuals([41.5], [1.5], [200], [2.0], [41.5, 41.5, 42.5], [1.5, 1.5, 1.5], [200, 1700, 200])
    assert abs(r[0] - 2.0) < 0.1          # junto a la estación
    assert r[1] == 0                       # 1500 m más alto: otra situación
    assert r[2] == 0                       # a 110 km


def test_palette_and_colorize():
    stops = palette_stops()
    assert stops[0][0] == -10 and stops[-1][0] == 40 and len(stops) == 9
    img = colorize(np.array([[np.nan, -10.0, 15.0, 40.0]]))
    assert img.shape == (1, 4, 4) and img[0, 0, 3] == 0 and (img[0, 1:, 3] == 255).all()
    assert tuple(img[0, 2, :3]) == (0xF0, 0xEF, 0xEC)
