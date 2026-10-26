from __future__ import annotations

YEAR = 2026

TESTS = [
    "Matemáticas",
    "Ciencias naturales",
    "Ciencias sociales",
    "Lenguaje",
    "Inglés",
]

SITES = ["BAQ", "COT", "MOS", "TUN", "USAQ", "ZIPA"]
ANTIGUEDAD_FULL_SITES = ["BAQ", "USAQ"]  # Portal de Génova y Usaquén
ANTIGUEDAD_SITES = ["BAQ", "COT", "MOS", "TUN", "USAQ", "ZIPA"]

LEVEL_ORDER = [
    "Nivel 1",
    "Nivel 2",
    "Nivel 3",
    "Nivel 4",
]

LEVEL_COLORS = {
    "Nivel 1": "#C93C3C",
    "Nivel 2": "#E58B2A",
    "Nivel 3": "#D8B531",
    "Nivel 4": "#2E8B57",
}

LEGACY_LEVEL_MAP = {
    "Progreso limitado": "Nivel 1",
    "Emergente": "Nivel 2",
    "En aceleración": "Nivel 3",
    "Avanzado": "Nivel 4",
}

LATAM_ORDER = ["Inicio", "Básico", "Satisfactorio", "Avanzado"]
LATAM_SCORE = {
    "Inicio": 1,
    "Básico": 2,
    "Satisfactorio": 3,
    "Avanzado": 4,
}

MATH_ACTIVITY_METRICS = [
    "Actividades realizadas",
    "Tiempo total (horas)",
    "Promedio general",
    "Numeración",
    "Relaciones y cambio",
    "Espacio y forma",
    "Medida",
    "Estadística y probabilidad",
]

MATH_COMPETENCIES = [
    "Numeración",
    "Relaciones y cambio",
    "Espacio y forma",
    "Medida",
    "Estadística y probabilidad",
]

EVOLUTION_METRICS = [
    "Puntaje global",
    "Puntaje Matemáticas",
    "Puntaje Lectura crítica",
    "Puntaje Ciencias naturales",
    "Puntaje Sociales y ciudadanas",
    "Puntaje Inglés",
]

IMPROVE_EFA_TESTS = [
    "Matemáticas",
    "Lenguaje",
    "Ciencias naturales",
    "Ciencias sociales",
    "Inglés",
]

GRADE_APPLICABILITY = {
    "Matemáticas": list(range(3, 12)),
    "Lenguaje": list(range(3, 12)),
    "Ciencias naturales": list(range(5, 12)),
    "Ciencias sociales": list(range(5, 12)),
    "Inglés": list(range(9, 12)),
}

# Regla mínima de divulgación para resultados públicos y visualizaciones.
MIN_PUBLIC_GROUP_SIZE = 5
