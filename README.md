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
