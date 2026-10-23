# Estado de llegada 2026

Sistema analítico en **Streamlit** para integrar resultados de aprendizaje de
Evaluar para Avanzar (EFA), evaluación interna, LATAM, actividad matemática /
Innovamat e Improve.

El proyecto evolucionó de un dashboard de una sola prueba hacia una herramienta
de **triangulación y trayectoria de aprendizaje**.

## Preguntas que responde

- ¿Cómo se distribuyen los estudiantes entre los cuatro niveles analíticos de EFA?
- ¿Qué sedes y grados concentran mayor porcentaje de estudiantes en ≤50%?
- ¿Cómo se comportan las dimensiones de cada prueba?
- ¿Qué patrones aparecen según antigüedad y colegio de origen?
- ¿Cómo se relacionan LATAM e Innovamat con EFA?
- ¿Cómo cambia Improve entre aplicaciones para los mismos estudiantes?
- ¿Qué puntaje EFA tienen estudiantes clasificados internamente como Progreso
  limitado, Emergente, En aceleración o Avanzado?
- ¿Cómo se ve una historia longitudinal agregada cuando se exige la máxima
  cobertura viable de fuentes?

## Fuentes y aplicabilidad

EFA se organiza en cinco pruebas canónicas:

| Prueba | Grados |
|---|---|
| Matemáticas | 3°–11° |
| Lenguaje | 3°–11° |
| Ciencias naturales | 5°–11° |
| Ciencias sociales | 5°–11° |
| Inglés | 9°–11° |

NIZA está excluida de la versión actual del análisis.

## Regla de puntuación EFA

```
Puntaje = Aciertos / Ítems esperados × 100
```

No se divide por el número de filas observadas. Una respuesta faltante puede no
existir como fila en la exportación, por lo que el denominador esperado protege
contra inflación artificial del puntaje.

| Prueba | Grado | Ítems esperados |
|---|---:|---:|
| Todas excepto Inglés | todos | 20 |
| Inglés | 9°–10° | 22 |
| Inglés | 11° | 25 |

## Niveles analíticos

| Banda | Rango |
|---|---|
| Nivel 1 | 0–25% |
| Nivel 2 | >25–50% |
| Nivel 3 | >50–75% |
| Nivel 4 | >75–100% |

Estas bandas son **internas y descriptivas**. No deben presentarse como niveles
oficiales de desempeño del ICFES ni como puntos de corte psicométricos externos.

## Vistas principales

1. **Análisis general**
2. **Análisis por antigüedad ≤1 año**
3. **LATAM vs Evaluar para Avanzar**
4. **Actividad matemática vs EFA**
5. **Evolución 10°–11°**
6. **EFA vs Improve**
7. **Interna vs EFA**
8. **Historia longitudinal**

Documentación complementaria:

- [Metodología analítica](docs/METHODOLOGY.md)
- [Pipeline de datos](docs/DATA_PIPELINE.md)
- [Arquitectura técnica](docs/ARCHITECTURE.md)

## Arquitectura

```text
epa_pablo/
├── app.py                  # navegación y vistas Streamlit
├── config.py               # configuración única del proyecto
├── processing.py           # cálculo base EFA
├── data_quality.py         # validación y control de divulgación
├── ui.py                   # utilidades UI reutilizables
├── data/                   # agregados públicos precalculados
├── docs/
│   └── METHODOLOGY.md
├── scripts/
│   └── audit_public_data.py
├── tests/
│   ├── test_processing.py
│   └── test_public_data.py
├── .github/workflows/
│   └── ci.yml
├── requirements.txt
└── requirements-dev.txt
```

`app.py` sigue siendo grande y su modularización por vistas es la principal
deuda técnica restante. La estrategia recomendada es mover una vista a la vez,
manteniendo tests de regresión.

## Calidad y privacidad

El repositorio contiene agregados precalculados para permitir una demo sin subir
bases originales con datos personales.

Reglas:

- no publicar nombres, correos, documentos o identificadores individuales;
- las historias longitudinales exigen al menos 5 estudiantes;
- las vistas sensibles suprimen grupos pequeños cuando es posible sin alterar
  denominadores;
- ejecutar auditoría de datos antes de cada despliegue.

Auditoría:

```bash
python scripts/audit_public_data.py
```

Modo estricto para detectar cualquier celda con n<5:

```bash
python scripts/audit_public_data.py --strict-small-cells
```

**Nota de gobierno:** el repositorio es actualmente público. Los agregados
históricos que contengan celdas pequeñas pueden persistir en el historial Git
aunque se eliminen de la rama actual. Para una operación institucional se
recomienda repositorio privado o una limpieza explícita del historial antes de
publicar una versión abierta.

## Pruebas automáticas

Instala dependencias de desarrollo:

```bash
pip install -r requirements-dev.txt
```

Ejecuta:

```bash
pytest -q
python scripts/audit_public_data.py
```

GitHub Actions ejecuta automáticamente:

- compilación de Python;
- auditoría de CSV públicos;
- tests de cálculo EFA;
- tests de privacidad/esquema;

en Python 3.12, 3.13 y 3.14.

## Despliegue

```bash
pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud, selecciona explícitamente la misma versión de
Python utilizada para validar el proyecto. La versión de Python se configura en
**Advanced settings** al desplegar.

## Principios de interpretación

El dashboard es adecuado para:

- revisión académica;
- generación de hipótesis;
- identificación de grupos que requieren apoyo;
- contraste entre instrumentos;
- seguimiento longitudinal agregado.

No debe utilizarse por sí solo para:

- inferir causalidad;
- evaluar el desempeño docente;
- clasificar formalmente estudiantes;
- tomar decisiones individuales de alto impacto.

## Estado del proyecto

La versión actual debe entenderse como un **piloto institucional avanzado**.
El foco de evolución ya no es agregar más visualizaciones, sino mejorar
reproducibilidad, automatización del pipeline, privacidad y modularidad.
