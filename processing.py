
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd


GRADE_MAP = {
    "PRIMERO": 1,
    "SEGUNDO": 2,
    "TERCERO": 3,
    "CUARTO": 4,
    "QUINTO": 5,
    "SEXTO": 6,
    "SEPTIMO": 7,
    "OCTAVO": 8,
    "NOVENO": 9,
    "DECIMO": 10,
    "UNDECIMO": 11,
}

OPTIONAL_COURSE_COLUMNS = [
    "Curso",
    "curso",
    "Grupo",
    "grupo",
    "Course",
    "course",
    "Salon",
    "Salón",
    "salon",
]


def _strip_accents(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", _strip_accents(value).strip()).upper()


def find_course_column(df: pd.DataFrame) -> Optional[str]:
    for col in OPTIONAL_COURSE_COLUMNS:
        if col in df.columns:
            return col
    normalized = {normalize_text(c): c for c in df.columns}
    for candidate in ["CURSO", "GRUPO", "SALON", "COURSE"]:
        if candidate in normalized:
            return normalized[candidate]
    return None


def grade_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    raw = normalize_text(value)
    if raw in GRADE_MAP:
        return float(GRADE_MAP[raw])
    match = re.search(r"\b(1[01]|[1-9])\b", raw)
    if match:
        return float(match.group(1))
    return np.nan


def normalize_area(quiz_name: object) -> str:
    """Homogeneiza el nombre de la prueba sin incluir el grado."""
    q = normalize_text(quiz_name)
    if "MATEM" in q:
        return "Matemáticas"
    if "LENGUAJE" in q or "LECTURA" in q:
        return "Lenguaje"
    if (
        "SOCIALES" in q
        or "CIUDADAN" in q
        or "PENSAMIENTO CIUDADANO" in q
    ):
        return "Ciencias sociales"
    if "CIENCIAS" in q:
        return "Ciencias naturales"
    if "INGLES" in q:
        return "Inglés"
    return str(quiz_name).strip()


def expected_items(area: object, grade_num: object, quiz_name: object = "") -> int:
    area_n = normalize_text(area)
    quiz_n = normalize_text(quiz_name)
    is_english = ("INGLES" in area_n) or ("INGLES" in quiz_n)
    try:
        grade = int(float(grade_num))
    except (TypeError, ValueError):
        grade = None

    if is_english and grade in (9, 10):
        return 22
    if is_english and grade == 11:
        return 25
    return 20


def performance_level(score: object) -> str:
    if pd.isna(score):
        return "Sin información"
    x = float(score)
    if x <= 25:
        return "Progreso limitado"
    if x <= 50:
        return "Emergente"
    if x <= 75:
        return "En aceleración"
    return "Avanzado"


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "si", "sí", "yes", "y", "correct", "correcto"})


def validate_columns(df: pd.DataFrame) -> list[str]:
    required = [
        "AttemptId",
        "Sede",
        "Grado",
        "IdentiEstudiante",
        "Nombre",
        "Apellido",
        "QuizName",
        "Pregunta",
        "IsCorrect",
    ]
    return [c for c in required if c not in df.columns]


def build_attempt_table(raw: pd.DataFrame) -> pd.DataFrame:
    missing = validate_columns(raw)
    if missing:
        raise ValueError(
            "Faltan columnas requeridas: " + ", ".join(missing)
        )

    df = raw.copy()
    df["QuizName_original"] = df["QuizName"]
    df["Grado_num"] = df["Grado"].apply(grade_number)
    df["Area"] = df["QuizName_original"].apply(normalize_area)
    df["EsCorrecta"] = _as_bool(df["IsCorrect"])

    course_col = find_course_column(df)
    if course_col:
        df["Curso"] = df[course_col].astype("string")
    else:
        df["Curso"] = pd.NA

    if "TimeCompleted" in df.columns:
        df["TimeCompleted"] = pd.to_datetime(df["TimeCompleted"], errors="coerce")
        df["Año"] = df["TimeCompleted"].dt.year.astype("Int64")
    else:
        df["Año"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # Una pregunta solo aporta un acierto máximo dentro del intento.
    # Esto protege el cálculo frente a duplicados accidentales de filas.
    keys = [
        "AttemptId", "IdentiEstudiante", "QuizName_original", "Pregunta"
    ]
    item_level = (
        df.groupby(keys, dropna=False, as_index=False)
        .agg(
            EsCorrecta=("EsCorrecta", "max"),
            Sede=("Sede", "first"),
            Grado=("Grado", "first"),
            Grado_num=("Grado_num", "first"),
            Area=("Area", "first"),
            Curso=("Curso", "first"),
            Nombre=("Nombre", "first"),
            Apellido=("Apellido", "first"),
            AntiguedadBS=("AntiguedadBS", "first") if "AntiguedadBS" in df.columns else ("Sede", lambda s: pd.NA),
            edad_estudiante=("edad_estudiante", "first") if "edad_estudiante" in df.columns else ("Sede", lambda s: pd.NA),
            colegio_de_origen=("colegio_de_origen", "first") if "colegio_de_origen" in df.columns else ("Sede", lambda s: pd.NA),
            TimeCompleted=("TimeCompleted", "max") if "TimeCompleted" in df.columns else ("Sede", lambda s: pd.NaT),
            Año=("Año", "first"),
        )
    )

    attempts = (
        item_level.groupby(
            ["AttemptId", "IdentiEstudiante", "QuizName_original"],
            dropna=False,
            as_index=False,
        )
        .agg(
            Sede=("Sede", "first"),
            Grado=("Grado", "first"),
            Grado_num=("Grado_num", "first"),
            Area=("Area", "first"),
            Curso=("Curso", "first"),
            Nombre=("Nombre", "first"),
            Apellido=("Apellido", "first"),
            AntiguedadBS=("AntiguedadBS", "first"),
            edad_estudiante=("edad_estudiante", "first"),
            colegio_de_origen=("colegio_de_origen", "first"),
            TimeCompleted=("TimeCompleted", "max"),
            Año=("Año", "first"),
            Aciertos=("EsCorrecta", "sum"),
            Items_observados=("Pregunta", "nunique"),
        )
    )

    attempts["Items_esperados"] = attempts.apply(
        lambda r: expected_items(r["Area"], r["Grado_num"], r["QuizName_original"]),
        axis=1,
    )
    attempts["QuizName"] = attempts["Area"]
    attempts["No_respondidos"] = (
        attempts["Items_esperados"] - attempts["Items_observados"]
    ).clip(lower=0)

    # La nota SIEMPRE usa el denominador esperado, no el número observado.
    attempts["Porcentaje_acierto"] = (
        attempts["Aciertos"] / attempts["Items_esperados"] * 100
    ).clip(lower=0, upper=100)

    attempts["Nivel_desempeno"] = attempts["Porcentaje_acierto"].apply(
        performance_level
    )
    attempts["Cobertura_respuesta"] = (
        attempts["Items_observados"] / attempts["Items_esperados"] * 100
    ).clip(lower=0, upper=100)

    attempts["Estudiante"] = (
        attempts["Nombre"].fillna("").astype(str).str.strip()
        + " "
        + attempts["Apellido"].fillna("").astype(str).str.strip()
    ).str.strip()

    attempts["Necesita_apoyo"] = attempts["Nivel_desempeno"].isin(
        ["Progreso limitado", "Emergente"]
    )
    attempts["Grupo_50"] = np.where(
        attempts["Porcentaje_acierto"] <= 50,
        "≤50%",
        ">50%",
    )

    return attempts



def build_dimension_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Construye resultados por dimensión/competencia a nivel de estudiante e intento.

    El denominador esperado de cada dimensión se estima como el máximo número de
    ítems observados para esa dimensión dentro de la misma prueba original.
    Esto evita inflar el porcentaje cuando una respuesta faltante no aparece
    como fila en la exportación.
    """
    missing = validate_columns(raw)
    if missing:
        raise ValueError(
            "Faltan columnas requeridas: " + ", ".join(missing)
        )
    if "competencia" not in raw.columns:
        return pd.DataFrame()

    df = raw.copy()
    df["QuizName_original"] = df["QuizName"]
    df["Grado_num"] = df["Grado"].apply(grade_number)
    df["Prueba"] = df["QuizName_original"].apply(normalize_area)
    df["EsCorrecta"] = _as_bool(df["IsCorrect"])
    df["Dimension"] = (
        df["competencia"]
        .astype("string")
        .str.strip()
    )

    if "TimeCompleted" in df.columns:
        df["TimeCompleted"] = pd.to_datetime(
            df["TimeCompleted"], errors="coerce"
        )
        df["Año"] = df["TimeCompleted"].dt.year.astype("Int64")
    else:
        df["Año"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    item_level = (
        df.groupby(
            [
                "AttemptId",
                "IdentiEstudiante",
                "QuizName_original",
                "Pregunta",
                "Dimension",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            EsCorrecta=("EsCorrecta", "max"),
            Sede=("Sede", "first"),
            Grado_num=("Grado_num", "first"),
            Prueba=("Prueba", "first"),
            Año=("Año", "first"),
        )
    )

    dimension_attempt = (
        item_level.groupby(
            [
                "AttemptId",
                "IdentiEstudiante",
                "QuizName_original",
                "Dimension",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            Sede=("Sede", "first"),
            Grado_num=("Grado_num", "first"),
            Prueba=("Prueba", "first"),
            Año=("Año", "first"),
            Aciertos_dimension=("EsCorrecta", "sum"),
            Items_observados_dimension=("Pregunta", "nunique"),
        )
    )

    expected = (
        dimension_attempt.groupby(
            ["QuizName_original", "Dimension"],
            dropna=False,
            as_index=False,
        )["Items_observados_dimension"]
        .max()
        .rename(
            columns={
                "Items_observados_dimension": "Items_esperados_dimension"
            }
        )
    )

    dimension_attempt = dimension_attempt.merge(
        expected,
        on=["QuizName_original", "Dimension"],
        how="left",
    )
    dimension_attempt["Porcentaje_dimension"] = (
        dimension_attempt["Aciertos_dimension"]
        / dimension_attempt["Items_esperados_dimension"]
        * 100
    ).clip(lower=0, upper=100)
    dimension_attempt["Nivel_dimension"] = (
        dimension_attempt["Porcentaje_dimension"].apply(performance_level)
    )

    return dimension_attempt


def aggregate_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame()

    out = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            Estudiantes=("IdentiEstudiante", "nunique"),
            Intentos=("AttemptId", "nunique"),
            Promedio=("Porcentaje_acierto", "mean"),
            Mediana=("Porcentaje_acierto", "median"),
            Cobertura_promedio=("Cobertura_respuesta", "mean"),
            Progreso_limitado=("Nivel_desempeno", lambda s: (s == "Progreso limitado").mean() * 100),
            Emergente=("Nivel_desempeno", lambda s: (s == "Emergente").mean() * 100),
            En_aceleracion=("Nivel_desempeno", lambda s: (s == "En aceleración").mean() * 100),
            Avanzado=("Nivel_desempeno", lambda s: (s == "Avanzado").mean() * 100),
            Requiere_apoyo=("Necesita_apoyo", "mean"),
        )
    )
    out["Requiere_apoyo"] *= 100
    return out
