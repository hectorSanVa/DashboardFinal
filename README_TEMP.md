# Dashboard Azúcar - Infocaña

Análisis predictivo de la producción de azúcar en México mediante regresión lineal y procesamiento ETL con datos abiertos de Infocaña.

## Descripción

Este dashboard fue desarrollado como parte del proyecto final de Big Data (UNACH, Ene-Jun 2026). Utiliza:

- **ETL**: Procesamiento de datos de múltiples zafras (2014-2025)
- **Regresión Lineal**: Predicción de producción de azúcar usando sklearn
- **Dashboard**: Shiny for Python con visualizaciones interactivas

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución local

```bash
python app.py
```

O simplemente:

```bash
shiny run app:app --reload
```

## Despliegue en GitHub Pages

Este proyecto está configurado para desplegarse en GitHub Pages usando ShinyLive.

## Estructura del Proyecto

```
├── app.py                  # Dashboard Shiny
├── etl/
│   └── pipeline.py        # Pipeline ETL
├── data/
│   ├── datos_procesados.csv
│   └── datos_agrupados_por_zafra.csv
├── csv/                   # Datos crudos de Infocaña
├── requirements.txt       # Dependencias
└── README.md
```

## Fuente de Datos

- https://www.datos.gob.mx/dataset/avance_produccion_cana_azucar_infocana