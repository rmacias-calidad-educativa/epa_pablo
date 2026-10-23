# Arquitectura

## Capas actuales

### config.py

Fuente única de configuración:

- año;
- sedes;
- pruebas;
- niveles;
- aplicabilidad por grado;
- tamaño mínimo de divulgación.

### processing.py

Lógica base de EFA y transformación de respuestas por ítem.

Esta capa no debe depender de Streamlit.

### data_quality.py

Utilidades de:

- validación de columnas;
- lectura validada de CSV;
- detección de identificadores;
- supresión de grupos pequeños.

### schemas.py

Contratos mínimos de los agregados precalculados.

### loaders.py

Carga, normalización y cache de los datos consumidos por la aplicación.

### view_*.py

Visualizaciones separadas por dominio:

- general;
- LATAM;
- actividad matemática;
- evolución;
- Improve;
- evaluación interna;
- historia longitudinal.

`views.py` actúa como fachada estable de importación.

### app.py

Capa ligera de navegación y composición de pestañas.

## Regla de dependencia

La dirección deseada es:

```text
config / schemas
       ↓
processing / data_quality
       ↓
loaders
       ↓
views
       ↓
app
```

Las capas inferiores no deben importar `app.py`.

## Estado de modularización

La capa de loaders y la capa de visualización ya fueron extraídas de
`app.py`.

La estructura actual mantiene una fachada `views.py` y módulos por dominio.
Esto permite modificar una vista sin intervenir en la navegación general.

La deuda técnica principal se desplaza ahora hacia la automatización completa
del ETL que genera los agregados de `data/`.

## Principios

1. Una regla de negocio debe vivir en un solo lugar.
2. Ninguna vista debe recalcular emparejamientos.
3. Los agregados públicos deben tener contrato explícito.
4. Las transformaciones deben ser testeables sin Streamlit.
5. Los errores de datos deben detectarse al cargar, no al graficar.
6. La privacidad se trata como parte del pipeline, no como decoración visual.
