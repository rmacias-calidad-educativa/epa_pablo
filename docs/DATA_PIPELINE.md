# Pipeline de datos

## Objetivo

Separar claramente:

1. **fuentes originales con datos individuales**;
2. **reglas de emparejamiento y transformación**;
3. **agregados públicos que consume Streamlit**.

Las fuentes originales no deben almacenarse en el repositorio público.

## Flujo general

```text
Fuentes originales
      │
      ├── EFA
      ├── Evaluación interna
      ├── LATAM
      ├── Innovamat / actividad matemática
      └── Improve
      │
      ▼
Normalización de identidad, grado, sede y prueba
      │
      ▼
Emparejamiento estudiante a estudiante
      │
      ▼
Cálculo de métricas y control de calidad
      │
      ▼
Agregación + control de divulgación
      │
      ▼
data/*.csv
      │
      ▼
loaders.py
      │
      ▼
app.py
```

## Evaluar para Avanzar

Fuente esperada a nivel de respuesta por ítem.

Transformaciones centrales en `processing.py`:

- normalización de grado;
- normalización del nombre de prueba;
- conversión de `IsCorrect`;
- deduplicación por intento + estudiante + prueba + pregunta;
- cálculo con denominador esperado;
- clasificación en Nivel 1–4.

## Evaluación interna

Llave de unión principal con EFA: **correo exacto normalizado**.

Validaciones posteriores:

- mismo grado;
- misma sede;
- área equivalente.

Mapeo curricular:

| EFA | Evaluación interna |
|---|---|
| Matemáticas | Innova Mate |
| Lenguaje | ABC Innova |
| Ciencias sociales | Global Citizens |
| Inglés | English Hub |
| Ciencias naturales 3°–9° | Scientific Inquiry |
| Ciencias naturales 10°–11° | Física + Química |

La vista temporal publicada utiliza Trimestre 1 y Trimestre 2.

## LATAM

Flujo de identidad:

```text
correo LATAM
→ correo en base puente de usuarios
→ identificador institucional
→ identificador EFA
```

La comparación publicada se limita a áreas equivalentes.

## Innovamat / actividad matemática

El cruce se realiza únicamente para Matemáticas.

Las celdas vacías de indicadores 0–10 son **NA**, no cero.
Los ceros registrados explícitamente sí son datos válidos.

## Improve

Emparejamiento estricto:

```text
documento validado + mismo grado + misma sede
```

Se excluyen registros marcados para revisión y no se usa fuzzy matching de
nombres para el análisis publicado.

Crecimiento longitudinal:

- 10°: EV1 → EV2.
- 11°: EV1 → EV3.

## Historia longitudinal

La cohorte se define por:

1. grado;
2. prueba EFA;
3. nivel EFA;
4. máxima cantidad viable de fuentes;
5. mínimo 5 estudiantes.

Los perfiles son agregados y usan medianas/promedios de la cohorte.

## Archivos derivados

Los principales agregados son cargados mediante `loaders.py`.

Sus contratos mínimos están declarados en `schemas.py`.

Si un archivo cambia de estructura, el loader debe fallar de forma temprana
mediante `read_csv_validated()`.

## Control de calidad previo a despliegue

```bash
python scripts/audit_public_data.py
pytest -q
```

El CI ejecuta estas validaciones automáticamente.

## Deuda de reproducibilidad

El repositorio todavía no contiene un pipeline ETL completo que regenere todos
los agregados desde las fuentes originales.

Para una versión institucional cerrada, el siguiente objetivo técnico debe ser
convertir cada cruce en una función o script determinista y ejecutar un único
comando de construcción, por ejemplo:

```bash
python scripts/build_all.py
```

Las fuentes originales deben residir fuera del repositorio público, por ejemplo
en almacenamiento institucional controlado.
