# meteo-cat

Predicción meteorológica para Catalunya a ~1 km. Combina varios modelos numéricos y corrige su error con machine learning entrenado con las estaciones de Meteocat y la orografía. Funciona solo con datos abiertos y gratuitos, y se ejecuta como servicio programado en GitHub Actions con el panel publicado en GitHub Pages.

**Estado: primera versión.** Predice la temperatura a 2 m de 0 a 72 h, con nowcasting de 0 a 6 h. Las demás variables se añadirán sobre la misma estructura.

## Fuentes de datos (todas gratuitas y sin clave)

| Fuente | Qué aporta | Acceso |
|---|---|---|
| ECMWF IFS 0,25° | Modelo global de referencia, pasos de 3 h | ECMWF Open Data (réplica en Google Cloud), histórico desde 2023 |
| ECMWF AIFS | Modelo de IA de ECMWF, pasos de 6 h | ECMWF Open Data, histórico desde 2024 |
| NOAA GFS 0,25° | Modelo global de EE. UU., pasos de 3 h | Bucket público de AWS, histórico desde 2021 |
| XEMA (Meteocat) | ~180 estaciones, observaciones horarias casi en tiempo real | Portal de datos abiertos de la Generalitat |
| Copernicus DEM GLO-90 | Orografía a 90 m | Bucket público de AWS |

De cada fichero GRIB global solo se descargan los campos necesarios (peticiones HTTP parciales, ~0,6 MB por campo) y se recortan a Catalunya.

AROME (Météo-France) e ICON-EU (DWD) también son abiertos, pero sin histórico público para entrenar; quedan para una fase posterior.

## Cómo funciona

1. **Modelos**: temperatura, punto de rocío, nubosidad y viento a 10 m de cada modelo, interpolados en el tiempo a cada hora y en el espacio a cada celda de ~1 km.
2. **Terreno**: altitud, rugosidad, pendiente, orientación, posición en valle o cresta (TPI), altitud a la escala de los modelos y distancia al mar.
3. **Base física**: media de los modelos más la corrección por altitud con el gradiente térmico estándar (-6,5 °C/km) respecto a la orografía suavizada a 25 km que ven los modelos.
4. **Corrector ML** (LightGBM): aprende el error de esa base con un año de previsiones pasadas y observaciones XEMA. Tiene en cuenta la hora, la época del año, el horizonte, la dispersión entre modelos, la humedad, la nubosidad, el viento y el terreno.
5. **Nowcasting** (0 a 6 h): el error actual en las estaciones se reparte por el territorio, penalizando las diferencias de altitud, y su peso decae con el horizonte según la persistencia medida en el entrenamiento.
6. **Panel**: mapa con barra de tiempo y evolución en cualquier punto, como web estática.

## Servicio programado (GitHub Actions)

| Flujo | Cuándo | Qué hace |
|---|---|---|
| `previsión` | Cada hora (minuto 17) | Descarga las últimas pasadas completas y las observaciones, genera la previsión y publica el panel en GitHub Pages |
| `reentrenamiento` | Día 2 de cada mes | Actualiza el histórico del último año (en la caché de Actions), reentrena y guarda el modelo en `models/` |
| `tests` | En cada cambio | Pruebas automáticas |

Requisitos del repositorio: ser público (minutos de Actions ilimitados y Pages gratuito) y tener GitHub Pages activado con origen "GitHub Actions" (Settings → Pages). GitHub desactiva los flujos programados de un repositorio público tras 60 días sin actividad; el reentrenamiento mensual hace un commit y lo evita.

## Verificación

`meteo-cat train` informa del error medio absoluto (MAE), el error cuadrático medio (RMSE) y el sesgo de cada modelo, de la media de modelos, de la base física y de la previsión corregida:

- en los últimos meses del histórico (validación temporal),
- por horizonte (día 0, 1 y 2),
- en estaciones reservadas que el modelo no ha visto, que simulan los puntos de la rejilla sin estación.

El último informe está en `models/report.txt`.

## Uso local

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
meteo-cat terrain                         # una vez: DEM y variables de terreno
meteo-cat history --start 2025-10-01      # histórico (reanudable, ~10 min por mes)
meteo-cat train                           # entrena y muestra la verificación
meteo-cat forecast --nowcast              # previsión
meteo-cat publish --out site              # web estática
meteo-cat serve --dir site                # verla en http://127.0.0.1:8000
```

Variables de entorno opcionales: `METEOCAT_DATA_DIR` (datos, por defecto `./var`), `METEOCAT_MODELS_DIR` (modelo entrenado, por defecto `./models`), `METEOCAT_MODELS`, `METEOCAT_FINE_GRID_STEP`, `METEOCAT_FORECAST_HOURS`, `SOCRATA_APP_TOKEN`.

## Próximas fases

1. Precipitación, humedad, viento y nubosidad con la misma estructura.
2. AROME e ICON-EU, acumulando su histórico desde ahora.
3. Satélite Meteosat (nubosidad, rayos MTG) y radar para el nowcasting de precipitación.
4. Temperatura superficial del mar (Copernicus Marine) para brisas y convección costera.
5. Interpolación por kriging con deriva orográfica en lugar de IDW.

## Licencias de datos

- ECMWF Open Data: CC BY 4.0, © ECMWF.
- NOAA GFS: dominio público.
- Observaciones XEMA: Servei Meteorològic de Catalunya, datos abiertos de la Generalitat.
- DEM: Copernicus GLO-90, © DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH 2014-2018, proporcionado bajo COPERNICUS por la Unión Europea y la ESA.
- Contorno de Catalunya: Natural Earth (dominio público).
- Mapa base: © OpenStreetMap, © CARTO.
