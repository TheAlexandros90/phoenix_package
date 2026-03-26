# Phoenix

Biblioteca de Python para limpiar, perfilar y preparar `pandas.DataFrame` de forma segura.

## Qué incluye

- Resúmenes de nulos, memoria y cardinalidad
- Reporte de calidad de datos con sugerencias
- Conversión segura a numérico, fecha, booleano y category
- Estandarización de nombres de columnas
- Extracción de texto, emails, URLs y fechas
- Detección de outliers
- Optimización de memoria

## Estructura del proyecto

```text
phoenix_package/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── phoenix/
│       ├── __init__.py
│       └── core.py
└── tests/
    └── test_phoenix.py
```

## Instalación en modo desarrollo

```bash
pip install -e .
```

## Uso básico

```python
import pandas as pd
from phoenix import Phoenix

df = pd.DataFrame({
    "cliente": [" Ana ", "Luis", None],
    "activo": ["si", "no", "verdadero"],
    "ventas": [100, 200, None],
})

phoenix = Phoenix(df)
phoenix.normalizar_texto(["cliente"], case="title")
phoenix.convertir_booleanos_texto(["activo"], allow_new_nulls=True)

print(phoenix.resumen_nulos())
print(phoenix.df)
```

## Ejecutar tests

```bash
python -m unittest discover -s tests -v
```

## Publicación en GitHub

1. Crea un repositorio nuevo.
2. Sube el contenido de esta carpeta.
3. Ajusta en `pyproject.toml` las URLs `TU_USUARIO`.
4. Opcionalmente añade una licencia si decides publicar el proyecto de forma abierta.