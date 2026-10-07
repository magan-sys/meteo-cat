"""Línea de comandos: meteo-cat <terrain|history|train|forecast|serve>."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="meteo-cat", description="Predicción meteorológica para Catalunya")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("terrain", help="Descarga el DEM y calcula las variables de terreno")

    h = sub.add_parser("history", help="Descarga observaciones XEMA y previsiones pasadas")
    h.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    h.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))

    sub.add_parser("train", help="Entrena el corrector y muestra la verificación")

    f = sub.add_parser("forecast", help="Genera la previsión en la rejilla de ~1 km")
    f.add_argument("--nowcast", action="store_true", help="Ajusta las primeras horas con observaciones recientes")

    s = sub.add_parser("serve", help="Arranca el panel web")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .config import get_settings
    settings = get_settings()

    if args.cmd == "terrain":
        from .grid import Grid
        from .terrain import build_terrain
        print(build_terrain(Grid(settings.fine_grid_step), settings.terrain_dir))

    elif args.cmd == "history":
        from .history import download
        download(settings, args.start, args.end)

    elif args.cmd == "train":
        from .grid import Grid
        from .history import assemble, station_terrain
        from .sources.openmeteo import HISTORY_VARS
        from .terrain import Terrain
        from .train import fit, format_report
        df = assemble(settings)
        terrain = Terrain(settings.terrain_dir / f"terrain_{Grid(settings.fine_grid_step).step:g}.npz")
        report = fit(df, station_terrain(df, terrain), settings.models, HISTORY_VARS, settings.models_dir)
        print(format_report(report))

    elif args.cmd == "forecast":
        from .forecast import Observations, Pipeline
        from .sources.openmeteo import OpenMeteoClient
        observations = None
        if args.nowcast:
            from .history import load_stations
            from .sources.xema import XemaOpenData
            today = datetime.now(timezone.utc).date()
            observations = Observations(load_stations(settings),
                                        XemaOpenData().hourly_observations(today - timedelta(days=1), today))
        meta = Pipeline(settings).run(OpenMeteoClient(settings.open_meteo_api_key), observations)
        print(json.dumps({k: meta[k] for k in ("issued", "ml_correction", "nowcast")}, indent=2))

    elif args.cmd == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
