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

### app.py

Navegación, composición de vistas y visualizaciones.

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

La capa de loaders ya fue extraída de `app.py`.

La siguiente deuda técnica es mover gradualmente las funciones de render a
módulos por dominio:

```text
views/
├── general.py
├── antiguedad.py
├── latam_efa.py
├── math_activity.py
├── evolution.py
├── improve_efa.py
├── internal_efa.py
└── story.py
```

La migración debe hacerse una vista a la vez, con CI verde después de cada
movimiento.

## Principios

1. Una regla de negocio debe vivir en un solo lugar.
2. Ninguna vista debe recalcular emparejamientos.
3. Los agregados públicos deben tener contrato explícito.
4. Las transformaciones deben ser testeables sin Streamlit.
5. Los errores de datos deben detectarse al cargar, no al graficar.
6. La privacidad se trata como parte del pipeline, no como decoración visual.
