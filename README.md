# Dashboard de estado de llegada

Aplicación en **Streamlit** para caracterizar el estado de llegada de estudiantes a partir de una exportación de respuestas por ítem.

## Qué resuelve

El dashboard permite analizar:

- niveles de desempeño por prueba;
- diferencias por sede y grado;
- curso/grupo cuando esa columna exista en la fuente;
- porcentaje de estudiantes en niveles que requieren mayor apoyo;
- cobertura de respuesta e ítems no observados;
- antigüedad en BS;
- colegio de origen;
- perfiles individuales por estudiante;
- auditoría del cálculo de cada intento.

## Regla de puntuación

La puntuación se calcula como:

`Aciertos / ítems esperados × 100`

**No** se divide por el número de filas observadas.

Esto es importante porque en la fuente una pregunta sin respuesta puede no aparecer. Por ejemplo, 8 aciertos con solo 10 preguntas observadas en una prueba de 20 ítems produce **40%**, no 80%.

### Ítems esperados

| Prueba | Grado | Ítems |
|---|---:|---:|
| Todas excepto Inglés | todos | 20 |
| Inglés | 9° | 22 |
| Inglés | 10° | 22 |
| Inglés | 11° | 25 |

### Niveles de desempeño

| Rango | Nivel |
|---|---|
| 0% a 25% | Progreso limitado |
| >25% a 50% | Emergente |
| >50% a 75% | En aceleración |
| >75% a 100% | Avanzado |

## Columnas esperadas

Obligatorias:

- `AttemptId`
- `Sede`
- `Grado`
- `IdentiEstudiante`
- `Nombre`
- `Apellido`
- `QuizName`
- `Pregunta`
- `IsCorrect`

Opcionales aprovechadas por el dashboard:

- `AntiguedadBS`
- `edad_estudiante`
- `colegio_de_origen`
- `TimeCompleted`
- `Curso`, `Grupo`, `Salón` o equivalentes

## Uso local

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Instala dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta:

```bash
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este proyecto a un repositorio de GitHub.
2. En Streamlit Community Cloud crea una nueva app.
3. Selecciona el repositorio.
4. Define `app.py` como archivo principal.
5. Despliega.

La app solicita el Excel mediante un cargador, por lo que **no es necesario subir al repositorio la base con datos personales**.

## Privacidad

El `.gitignore` excluye archivos Excel y el contenido de `data/`. Mantén las bases con nombres, identificaciones y colegios de origen fuera del repositorio público.

## Estructura

```text
dashboard_estado_llegada/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── processing.py
    └── ui.py
```


## Lecturas principales del dashboard

La versión actual prioriza dos preguntas de negocio educativo:

1. **¿Cuántos estudiantes están en ≤50% y cuántos en >50% en cada sede y prueba?**
   - Matriz con las seis sedes siempre visibles.
   - Filtros por año y grado.
   - Conteos dentro de cada celda.
   - Histórico anual por prueba y sede.

2. **¿Existen patrones relacionados con el colegio de origen?**
   - Distribución por los cuatro niveles de desempeño.
   - Comparación colegio de origen × área.
   - Diferencia del %≤50 frente al promedio de la red.
   - Porcentaje de estudiantes con dos o más pruebas en ≤50%.
   - Identificación descriptiva de patrones recurrentes en varias áreas.

Los patrones son **descriptivos, no causales**, y deben interpretarse junto con el número de estudiantes representados.

### Nota sobre años disponibles

La fuente actual contiene registros de 2025 y 2026, pero 2025 tiene una cantidad muy reducida de observaciones frente a 2026. La aplicación muestra una advertencia cuando se consulta ese periodo.


## Taxonomía de pruebas

Para la visualización analítica, los nombres de prueba se homogeneizan y el grado se trata como una dimensión independiente:

- **Ciencias naturales**: agrupa los registros etiquetados como Ciencias.
- **Ciencias sociales**: agrupa Competencias ciudadanas, Pensamiento ciudadano y Sociales y ciudadanas.
- **Matemáticas**: agrupa todas las pruebas de Matemáticas.
- **Lenguaje**: agrupa todas las pruebas de Lenguaje.

El grado no aparece dentro del rótulo de la prueba. Por ejemplo, `MATEMÁTICAS 7°` se visualiza como **Matemáticas**, mientras el grado 7° aparece en el eje correspondiente.

La visual principal es una gráfica de líneas donde:
- eje X = grado;
- eje Y = número de estudiantes con ≤50% de aciertos;
- cada línea = una de las seis sedes/colegios;
- filtro = una prueba a la vez, con opción de año;
- tooltip = estudiantes ≤50%, estudiantes >50%, total y porcentaje ≤50%.
