import numpy as np
import pandas as pd

from meteo_cat.grid import Grid, catalunya_mask
from meteo_cat.interp import LinearInterpolator, idw_residuals
from meteo_cat.render import colorize, palette_stops
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


def test_decode_crop_from_grib():
    import eccodes
    from meteo_cat.sources.nwp import convert_units, decode_crop

    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    # Rejilla global 1° que empieza en lon 180 (como ECMWF Open Data) y escanea de norte a sur.
    for k, v in {"Ni": 360, "Nj": 181, "latitudeOfFirstGridPointInDegrees": 90.0,
                 "longitudeOfFirstGridPointInDegrees": 180.0, "latitudeOfLastGridPointInDegrees": -90.0,
                 "longitudeOfLastGridPointInDegrees": 179.0, "iDirectionIncrementInDegrees": 1.0,
                 "jDirectionIncrementInDegrees": 1.0, "jScansPositively": 0}.items():
        eccodes.codes_set(h, k, v)
    lat = 90.0 - np.arange(181)
    lon = (180.0 + np.arange(360) + 180.0) % 360.0 - 180.0
    field = 273.15 + lat[:, None] * 0.1 + lon[None, :] * 0.01
    eccodes.codes_set_values(h, field.ravel())
    blob = eccodes.codes_get_message(h)
    eccodes.codes_release(h)

    lats, lons, crop, _ = decode_crop(blob)
    assert lats[0] > lats[-1] and np.all(np.diff(lons) > 0)
    assert lats.min() >= 39.75 and lats.max() <= 43.75 and lons.min() >= -0.75 and lons.max() <= 4.25
    i, j = list(lats).index(41.0), list(lons).index(2.0)
    assert abs(crop[i, j] - (273.15 + 4.1 + 0.02)) < 1e-3
    assert abs(convert_units("t2m", crop, "gfs")[i, j] - 4.12) < 1e-3
    assert convert_units("tcc", np.array([0.5]), "ecmwf")[0] == 50.0
    assert convert_units("tcc", np.array([0.5]), "gfs")[0] == 0.5
    assert convert_units("tp", np.array([0.002]), "ecmwf", "m")[0] == 2.0
    assert convert_units("tp", np.array([2.0]), "ecmwf", "kg m**-2")[0] == 2.0


def test_gfs_index_selects_instantaneous_fields():
    from meteo_cat.sources.nwp import NWPClient

    idx = "\n".join([
        "580:417000000:d=2026090100:HGT:surface:24 hour fcst:",
        "581:417829203:d=2026090100:TMP:2 m above ground:24 hour fcst:",
        "583:419563246:d=2026090100:DPT:2 m above ground:24 hour fcst:",
        "636:446191656:d=2026090100:TCDC:entire atmosphere:18-24 hour ave fcst:",
        "637:447021373:d=2026090100:TCDC:entire atmosphere:24 hour fcst:",
        "638:447900000:d=2026090100:PRES:surface:24 hour fcst:",
        "639:448000000:d=2026090100:APCP:surface:18-24 hour acc fcst:",
        "640:448100000:d=2026090100:APCP:surface:0-1 day acc fcst:",
        "641:448200000:d=2026090100:CPOFP:surface:24 hour fcst:",
    ]).encode()

    class Client(NWPClient):
        def _get(self, url, rng=None):
            return idx

    from datetime import datetime, timezone
    url, ranges = Client()._gfs_ranges(datetime(2026, 9, 1, tzinfo=timezone.utc), 24, ("t2m", "d2m", "tcc", "tp"))
    assert url.endswith("gfs.20260901/00/atmos/gfs.t00z.pgrb2.0p25.f024")
    assert ranges["t2m"] == (417829203, 419563245)
    assert ranges["tcc"] == (447021373, 447899999)  # la instantánea, no la media 18-24 h
    assert ranges["tp"] == (448100000, 448199999)   # el acumulado desde el inicio


def test_interp_steps_linear_in_time():
    from datetime import datetime, timezone
    from meteo_cat.history import interp_steps
    from meteo_cat.sources.nwp import RunFields

    arr = np.stack([np.full((2, 2), v, "float32") for v in (0.0, 6.0, 12.0)])
    run = RunFields("m", datetime(2026, 1, 1, tzinfo=timezone.utc), np.array([0, 6, 12]),
                    np.array([42.0, 41.75]), np.array([1.0, 1.25]), {"t2m": arr})
    out = interp_steps(run, [0, 3, 6, 9, 12, 15])["t2m"][:, 0, 0]
    assert np.allclose(out[:5], [0, 3, 6, 9, 12]) and np.isnan(out[5])


def test_xema_precipitation_3h_windows():
    from datetime import date
    from meteo_cat.sources.xema import sum_3h

    times = pd.date_range("2026-09-20T00:00", "2026-09-20T05:30", freq="30min")
    df = pd.DataFrame({"codi_estacio": "A", "data_lectura": times.strftime("%Y-%m-%dT%H:%M:%S.000"),
                       "valor_lectura": ["0.1"] * 6 + ["1.0"] * 6})
    df = pd.concat([df, pd.DataFrame({"codi_estacio": ["B"] * 3, "valor_lectura": ["5"] * 3,
                                      "data_lectura": times[:3].strftime("%Y-%m-%dT%H:%M:%S.000")})])
    out = sum_3h(df, date(2026, 9, 20), date(2026, 9, 20)).set_index(["station", "time"])["ppt3"]
    # Ventana (00:00, 03:00]: lecturas de 00:00 a 02:30; (03:00, 06:00]: de 03:00 a 05:30.
    assert abs(out[("A", pd.Timestamp("2026-09-20T03:00Z"))] - 0.6) < 1e-9
    assert abs(out[("A", pd.Timestamp("2026-09-20T06:00Z"))] - 6.0) < 1e-9
    assert ("B", pd.Timestamp("2026-09-20T03:00Z")) not in out.index  # ventana incompleta
