# Rregresion_logistica

Proyecto académico de Estadística Aplicada: generación de un conjunto de datos sintético y análisis de una regresión logística binaria para explorar el incumplimiento crediticio.

> **Alcance:** todos los registros son sintéticos y reproducibles; no representan personas ni clientes reales. Los resultados son demostrativos y no deben usarse para decisiones crediticias.

## Contenido

- `logistica.py`: genera los datos, estima e interpreta modelos logísticos, crea gráficas y construye el informe DOCX.
- `datos_credito_sinteticos.csv`: conjunto sintético generado por el script (2,400 registros, 10 predictores numéricos y una respuesta binaria).
- `figura_01_frecuencia_incumplimiento.png`: frecuencia de la variable dependiente.
- `figura_02_curva_logistica.png`: curva logística y observaciones de prueba.
- `figura_03_matriz_confusion.png`: matriz de confusión del modelo seleccionado.
- `requirements.txt`: dependencias de Python.

Al ejecutar el script se vuelven a generar el CSV, las tres figuras y el informe:
`ACTIVIDAD-5-ESTADISTICA-APLICADA-SEP-DIC-2026-ENTREGABLE.docx`.

## Requisitos

- Python 3.11 o posterior
- Paquetes enumerados en `requirements.txt`

## Ejecución

Desde la carpeta del proyecto, crea y activa un entorno virtual e instala las dependencias:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python logistica.py
```

El programa usa una semilla fija (`20261120`), así que los datos y resultados se pueden reproducir. El análisis divide los registros en entrenamiento y prueba de forma estratificada, estandariza los predictores usando únicamente entrenamiento y compara modelos de hasta cuatro variables con los criterios de p-value LLR y pseudo R-cuadrado de McFadden.

## Resultados de referencia

Con los datos generados por el script:

- Incumplimiento simulado: 23.0%.
- Pseudo R-cuadrado de McFadden del modelo completo: 0.25003.
- Modelo reducido seleccionado por LLR: `razon_deuda`, `utilizacion_credito`, `atrasos_12m` y `ahorro_mensual`.
- Exactitud de prueba del modelo reducido: 80.0%.
- Sensibilidad de prueba del modelo reducido: 27.5%.

La exactitud por sí sola puede ocultar errores importantes. La sensibilidad indica que el modelo seleccionado no detecta una parte considerable de los incumplimientos simulados; por tanto, estas métricas ilustran el análisis y no acreditan un modelo listo para uso real.

## Reproducibilidad y privacidad

No se descargan datos de Kaggle ni se incluyen datos personales. El CSV es generado localmente por `logistica.py`. Los integrantes deben completar los campos pendientes de la portada del informe antes de entregar el trabajo.
