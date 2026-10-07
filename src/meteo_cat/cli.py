"""Línea de comandos: meteo-cat <terrain|history|train|forecast|publish|serve>."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="meteo-cat", description="Predicción meteorológica para Catalunya")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("terrain", help="Descarga el DEM y calcula las variables de terreno")

    h = sub.add_parser("history", help="Descarga observaciones XEMA y previsiones pasadas de los modelos")
    h.add_argument("--start", type=date.fromisoformat, default=date.today() - timedelta(days=365))
    h.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=4))
    h.add_argument("--what", choices=("all", "temp", "precip"), default="all")

    tr = sub.add_parser("train", help="Entrena los correctores y muestra la verificación")
    tr.add_argument("--what", choices=("all", "temp", "precip"), default="all")

    f = sub.add_parser("forecast", help="Genera la previsión en la rejilla de ~1 km")
    f.add_argument("--nowcast", action="store_true", help="Ajusta las primeras horas con observaciones recientes")

    pb = sub.add_parser("publish", help="Genera el panel como web estática")
    pb.add_argument("--out", type=Path, default=Path("site"))

    s = sub.add_parser("serve", help="Sirve la web estática en local")
    s.add_argument("--dir", type=Path, default=Path("site"))
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
        download(settings, args.start, args.end, ("temp", "precip") if args.what == "all" else (args.what,))

    elif args.cmd == "train":
        from .grid import Grid
        from .history import assemble, assemble_precip, station_terrain
        from .terrain import Terrain
        from .train import fit, fit_rain, format_rain_report, format_report
        terrain = Terrain(settings.terrain_dir / f"terrain_{Grid(settings.fine_grid_step).step:g}.npz")
        if args.what in ("all", "temp"):
            df = assemble(settings)
            print(format_report(fit(df, station_terrain(df, terrain), settings.models, settings.models_dir)))
            print()
        if args.what in ("all", "precip"):
            df = assemble_precip(settings)
            print(format_rain_report(fit_rain(df, station_terrain(df, terrain), settings.models, settings.models_dir)))

    elif args.cmd == "forecast":
        from .forecast import Observations, Pipeline
        from .sources.nwp import NWPClient
        observations = None
        if args.nowcast:
            from .history import load_stations
            from .sources.xema import XemaOpenData
            today = datetime.now(timezone.utc).date()
            xema = XemaOpenData()
            observations = Observations(load_stations(settings),
                                        xema.hourly_observations(today - timedelta(days=1), today),
                                        xema.precipitation_3h(today - timedelta(days=1), today))
        meta = Pipeline(settings).run(NWPClient(), observations)
        print(json.dumps({k: meta[k] for k in ("issued", "runs", "ml_correction", "ml_rain", "nowcast")}, indent=2))

    elif args.cmd == "publish":
        from .publish import publish
        print(publish(settings, args.out))

    elif args.cmd == "serve":
        import functools
        import http.server
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.dir))
        print(f"http://127.0.0.1:{args.port}/")
        http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
