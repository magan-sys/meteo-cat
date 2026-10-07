# meteo-cat

Predicción meteorológica para Catalunya a ~1 km. Combina varios modelos numéricos y corrige su error con machine learning entrenado con las estaciones de Meteocat y la orografía.

**Estado: primera versión.** Predice la temperatura a 2 m de 0 a 72 h, con nowcasting de 0 a 6 h. Las demás variables se añadirán sobre la misma estructura.

## Cómo funciona

1. **Modelos numéricos** (Open-Meteo): AROME HD, ARPEGE, ICON-EU y ECMWF IFS (ICON-D2 no cubre Catalunya). Se piden sin la corrección de altitud de Open-Meteo, porque el ajuste por orografía lo hace este modelo.
2. **Terreno** (DEM Copernicus GLO-90): altitud, rugosidad, pendiente, orientación, posición en valle o cresta (TPI), altitud a la escala de los modelos y distancia al mar.
3. **Base física**: media de los modelos más la corrección por altitud con el gradiente térmico estándar (-6,5 °C/km).
4. **Corrector ML** (LightGBM): aprende el error de esa base a partir del histórico de previsiones pasadas (Open-Meteo Previous Runs) y las observaciones de la XEMA. Tiene en cuenta la hora, la época del año, el horizonte, la dispersión entre modelos, la nubosidad, el viento y el terreno.
5. **Nowcasting** (0 a 6 h): el error actual en las estaciones se reparte por el territorio, penalizando las diferencias de altitud, y su peso decae con el horizonte. La escala de decaimiento (tau) se mide en el entrenamiento.
6. **Rejilla de ~1 km** (0,01°) recortada a Catalunya, con un panel web: mapa con barra de tiempo y evolución en cualquier punto.

## Verificación

`meteo-cat train` informa del error medio absoluto (MAE), el error cuadrático medio (RMSE) y el sesgo de cada modelo, de la media de modelos, de la base física y de la previsión corregida:

- en los últimos meses del histórico (validación temporal),
- por horizonte (día 0, 1 y 2),
- en estaciones reservadas que el modelo no ha visto, que simulan los puntos de la rejilla sin estación.

También comprueba si las horas de las observaciones encajan con las de los modelos (detecta errores de zona horaria).

## Instalación

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # y rellenar lo necesario
```

## Uso

```bash
meteo-cat terrain                         # una vez: DEM y variables de terreno (~35 MB)
meteo-cat history --start 2024-01-01      # histórico de observaciones y previsiones (reanudable)
meteo-cat train                           # entrena y muestra la verificación
meteo-cat forecast --nowcast              # genera la previsión (programar cada 1-3 h)
meteo-cat serve --host 0.0.0.0            # panel en http://localhost:8000
```

Sin modelo entrenado, `forecast` funciona igual con la base física (media de modelos + altitud).

Ejemplo de programación con cron (cada 3 horas):

```
15 */3 * * * cd /ruta/meteo-cat && .venv/bin/meteo-cat forecast --nowcast >> var/forecast.log 2>&1
```

## Configuración (variables de entorno)

| Variable | Uso |
|---|---|
| `METEOCAT_DATA_DIR` | Carpeta de datos (por defecto `./var`) |
| `OPEN_METEO_API_KEY` | Clave comercial de Open-Meteo. **El uso gratuito es solo no comercial.** |
| `METEOCAT_API_KEY` | API XEMA de Meteocat (próxima fase: observaciones en tiempo real) |
| `AEMET_API_KEY` | AEMET OpenData (próxima fase) |
| `SOCRATA_APP_TOKEN` | Opcional, amplía los límites del portal de datos abiertos |
| `METEOCAT_MODELS` | Modelos separados por comas |
| `METEOCAT_MODEL_GRID_STEP` | Malla de consulta de modelos en grados (0,1 por defecto; 0,05 aprovecha mejor AROME pero cuesta 4 veces más llamadas) |
| `METEOCAT_FINE_GRID_STEP` | Rejilla final (0,01 por defecto) |
| `METEOCAT_FORECAST_HOURS` | Horizonte (72 por defecto) |

## Pendiente de verificar con datos reales

Este código se desarrolló sin acceso de red a Open-Meteo ni al portal de datos abiertos de la Generalitat. Está probado con datos sintéticos (`pytest`), y el terreno con el DEM real. En la primera ejecución real hay que confirmar:

- Zona horaria de `data_lectura` (se asume UTC; la última lectura publicada va menos de 1 h por detrás de la hora UTC, lo que apunta a UTC). `train` avisa si detecta un desfase.
- Fecha desde la que la API Previous Runs tiene datos `previous_dayN` de cada modelo (en enero de 2024 todavía no; en septiembre de 2026 sí).

Confirmado con datos reales: campos de los conjuntos XEMA, código de temperatura `32`, publicación casi en tiempo real y formato de las claves `<variable>_previous_dayN_<modelo>`.

## Próximas fases

1. API XEMA en tiempo real (nowcasting operativo) y estaciones de AEMET.
2. Precipitación, humedad, viento y nubosidad con la misma estructura.
3. Satélite Meteosat (nubosidad, rayos MTG) y radar para el nowcasting de precipitación.
4. Temperatura superficial del mar (Copernicus Marine) para brisas y convección costera.
5. Interpolación por kriging con deriva orográfica en lugar de IDW.

## Licencias de datos

- Contorno de Catalunya: Natural Earth (dominio público).
- DEM: Copernicus GLO-90, © DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH 2014-2018, proporcionado bajo COPERNICUS por la Unión Europea y la ESA.
- Observaciones XEMA: Servei Meteorològic de Catalunya, datos abiertos de la Generalitat.
- Modelos: Open-Meteo (CC BY 4.0) y los servicios meteorológicos de origen.
