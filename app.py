
from __future__ import annotations

import base64
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from processing import (
    build_attempt_table,
    build_dimension_table,
    normalize_area,
)


YEAR = 2026
TESTS = [
    "Matemáticas",
    "Ciencias naturales",
    "Ciencias sociales",
    "Lenguaje",
    "Inglés",
]
SITES = ["BAQ", "COT", "MOS", "TUN", "USAQ", "ZIPA"]
ANTIGUEDAD_SITES = ["COT", "MOS", "TUN", "ZIPA"]
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
LEGACY_LEVEL_MAP = {
    "Progreso limitado": "Nivel 1",
    "Emergente": "Nivel 2",
    "En aceleración": "Nivel 3",
    "Avanzado": "Nivel 4",
}


st.set_page_config(
    page_title="Estado de llegada 2026",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        max-width: 1550px;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.16);
        padding: 10px 12px;
        border-radius: 12px;
    }
    .threshold-box {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.18);
        margin: 8px 0 18px 0;
        font-size: 1.02rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_latam_efa_data():
    detail_path = Path("data/latam_efa_aggregate_2026.csv")
    summary_path = Path("data/latam_efa_match_summary_2026.csv")

    detail = (
        pd.read_csv(detail_path)
        if detail_path.exists()
        else pd.DataFrame()
    )
    summary = (
        pd.read_csv(summary_path)
        if summary_path.exists()
        else pd.DataFrame()
    )

    if not detail.empty:
        detail = detail[
            detail["Sede"].astype(str).isin(SITES)
        ].copy()
        detail["Grado_num"] = pd.to_numeric(
            detail["Grado_num"], errors="coerce"
        ).astype("Int64")
        detail["Pct_EFA"] = pd.to_numeric(
            detail["Pct_EFA"], errors="coerce"
        )
        detail["Estudiantes"] = pd.to_numeric(
            detail["Estudiantes"], errors="coerce"
        ).fillna(0).astype(int)

    return detail, summary


@st.cache_data(show_spinner=False)
def read_math_activity_data():
    stats_path = Path("data/math_activity_stats_2026.csv")
    bins_path = Path("data/math_activity_bins_2026.csv")
    summary_path = Path("data/math_activity_match_summary_2026.csv")

    stats = (
        pd.read_csv(stats_path)
        if stats_path.exists()
        else pd.DataFrame()
    )
    bins = (
        pd.read_csv(bins_path)
        if bins_path.exists()
        else pd.DataFrame()
    )
    summary = (
        pd.read_csv(summary_path)
        if summary_path.exists()
        else pd.DataFrame()
    )

    for df in [stats, bins]:
        if not df.empty:
            df["Grado_num"] = pd.to_numeric(
                df["Grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["Grado_num"], inplace=True)
            df["Grado_num"] = df["Grado_num"].astype(int)
            df.drop(
                df[~df["Sede"].astype(str).isin(SITES)].index,
                inplace=True,
            )

    return stats, bins, summary


@st.cache_data(show_spinner=False)
def read_evolution_10_11_data():
    summary_path = Path("data/evolucion_10_11_summary_2026.csv")
    trajectory_path = Path("data/evolucion_11_trayectoria_2026.csv")
    availability_path = Path(
        "data/evolucion_10_11_disponibilidad_2026.csv"
    )
    observed_path = Path(
        "data/evolucion_10_11_observado_2026.csv"
    )

    summary = (
        pd.read_csv(summary_path)
        if summary_path.exists()
        else pd.DataFrame()
    )
    trajectory = (
        pd.read_csv(trajectory_path)
        if trajectory_path.exists()
        else pd.DataFrame()
    )
    availability = (
        pd.read_csv(availability_path)
        if availability_path.exists()
        else pd.DataFrame()
    )
    observed = (
        pd.read_csv(observed_path)
        if observed_path.exists()
        else pd.DataFrame()
    )

    for df in [summary, trajectory, availability, observed]:
        if not df.empty and "Grado_num" in df.columns:
            df["Grado_num"] = pd.to_numeric(
                df["Grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["Grado_num"], inplace=True)
            df["Grado_num"] = df["Grado_num"].astype(int)

    return summary, trajectory, availability, observed


@st.cache_data(show_spinner=False)
def read_improve_efa_data():
    levels_path = Path("data/improve_efa_levels_2026.csv")
    quality_path = Path("data/improve_efa_match_quality_2026.csv")
    growth_path = Path("data/improve_growth_by_efa_level_2026.csv")

    levels = (
        pd.read_csv(levels_path)
        if levels_path.exists()
        else pd.DataFrame()
    )
    quality = (
        pd.read_csv(quality_path)
        if quality_path.exists()
        else pd.DataFrame()
    )
    growth = (
        pd.read_csv(growth_path)
        if growth_path.exists()
        else pd.DataFrame()
    )

    for df in [levels, quality, growth]:
        if not df.empty and "grado_num" in df.columns:
            df["grado_num"] = pd.to_numeric(
                df["grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["grado_num"], inplace=True)
            df["grado_num"] = df["grado_num"].astype(int)

    if not levels.empty:
        levels["Orden"] = pd.to_numeric(
            levels["Orden"], errors="coerce"
        )
        levels["n"] = pd.to_numeric(
            levels["n"], errors="coerce"
        ).fillna(0).astype(int)
        levels["Promedio_EFA"] = pd.to_numeric(
            levels["Promedio_EFA"], errors="coerce"
        )
        levels["Promedio_Improve"] = pd.to_numeric(
            levels["Promedio_Improve"], errors="coerce"
        )

    if not growth.empty:
        for col in [
            "Orden",
            "n",
            "Promedio_EFA",
            "Improve_inicial",
            "Improve_final",
            "Cambio_Improve",
            "Pct_mejora",
            "Pct_baja",
        ]:
            growth[col] = pd.to_numeric(
                growth[col], errors="coerce"
            )
        growth["n"] = growth["n"].fillna(0).astype(int)

    return levels, quality, growth


@st.cache_data(show_spinner=False)
def read_internal_efa_data():
    summary_path = Path("data/internal_efa_summary_2026.csv")
    distribution_path = Path("data/internal_efa_distribution_2026.csv")
    coverage_path = Path("data/internal_efa_coverage_2026.csv")
    agreement_path = Path("data/internal_efa_agreement_2026.csv")
    reverse_path = Path("data/internal_efa_reverse_t1_t2_2026.csv")

    summary = (
        pd.read_csv(summary_path)
        if summary_path.exists()
        else pd.DataFrame()
    )
    distribution = (
        pd.read_csv(distribution_path)
        if distribution_path.exists()
        else pd.DataFrame()
    )
    coverage = (
        pd.read_csv(coverage_path)
        if coverage_path.exists()
        else pd.DataFrame()
    )
    agreement = (
        pd.read_csv(agreement_path)
        if agreement_path.exists()
        else pd.DataFrame()
    )
    reverse = (
        pd.read_csv(reverse_path)
        if reverse_path.exists()
        else pd.DataFrame()
    )

    for df in [summary, distribution, coverage, agreement, reverse]:
        if not df.empty and "Grado_num" in df.columns:
            df["Grado_num"] = pd.to_numeric(
                df["Grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["Grado_num"], inplace=True)
            df["Grado_num"] = df["Grado_num"].astype(int)

    if not summary.empty:
        summary["Nivel_EFA_num"] = pd.to_numeric(
            summary["Nivel_EFA_num"], errors="coerce"
        )
        summary["Estudiantes"] = pd.to_numeric(
            summary["Estudiantes"], errors="coerce"
        ).fillna(0).astype(int)
        summary["Promedio_interno"] = pd.to_numeric(
            summary["Promedio_interno"], errors="coerce"
        )
        summary["Promedio_EFA"] = pd.to_numeric(
            summary["Promedio_EFA"], errors="coerce"
        )

    if not distribution.empty:
        distribution["Nivel_EFA_num"] = pd.to_numeric(
            distribution["Nivel_EFA_num"], errors="coerce"
        )
        distribution["Estudiantes"] = pd.to_numeric(
            distribution["Estudiantes"], errors="coerce"
        ).fillna(0).astype(int)
        distribution["Porcentaje"] = pd.to_numeric(
            distribution["Porcentaje"], errors="coerce"
        )

    if not coverage.empty:
        for col in ["EFA_estudiantes", "Emparejados"]:
            coverage[col] = pd.to_numeric(
                coverage[col], errors="coerce"
            ).fillna(0).astype(int)
        coverage["Cobertura_pct"] = pd.to_numeric(
            coverage["Cobertura_pct"], errors="coerce"
        )

    if not agreement.empty:
        for col in [
            "Coincidencia_exacta_pct",
            "Dentro_un_nivel_pct",
            "Promedio_interno",
            "Promedio_nivel_EFA",
        ]:
            agreement[col] = pd.to_numeric(
                agreement[col], errors="coerce"
            )

    if not reverse.empty:
        for col in [
            "grado_num",
            "nivel_num",
            "Estudiantes",
            "Promedio_EFA",
            "Mediana_EFA",
            "DE_EFA",
            "Trimestre_orden",
        ]:
            reverse[col] = pd.to_numeric(
                reverse[col], errors="coerce"
            )
        reverse["grado_num"] = reverse[
            "grado_num"
        ].astype("Int64")
        reverse.dropna(subset=["grado_num"], inplace=True)
        reverse["grado_num"] = reverse["grado_num"].astype(int)
        reverse["Estudiantes"] = reverse[
            "Estudiantes"
        ].fillna(0).astype(int)

    return summary, distribution, coverage, agreement, reverse


@st.cache_data(show_spinner=False)
def read_student_story_profiles():
    path = Path("data/student_story_profiles_aggregate_2026.csv")
    data = (
        pd.read_csv(path)
        if path.exists()
        else pd.DataFrame()
    )

    if not data.empty:
        data["grado_num"] = pd.to_numeric(
            data["grado_num"], errors="coerce"
        ).astype("Int64")
        data.dropna(subset=["grado_num"], inplace=True)
        data["grado_num"] = data["grado_num"].astype(int)
        data["Candidatos_completos"] = pd.to_numeric(
            data["Candidatos_completos"], errors="coerce"
        ).fillna(0).astype(int)

        for col in data.columns:
            if (
                col.startswith("EFA_")
                or col.startswith("Improve_")
            ) and col not in [
                "Prueba_filtro",
                "Nivel_EFA_filtro",
            ]:
                data[col] = pd.to_numeric(
                    data[col], errors="coerce"
                )

    return data


@st.cache_data(show_spinner=False)
def read_student_story_3_9():
    path = Path("data/student_story_3_9_aggregate_2026.csv")
    data = (
        pd.read_csv(path)
        if path.exists()
        else pd.DataFrame()
    )

    if not data.empty:
        data["grado_num"] = pd.to_numeric(
            data["grado_num"], errors="coerce"
        ).astype("Int64")
        data.dropna(subset=["grado_num"], inplace=True)
        data["grado_num"] = data["grado_num"].astype(int)
        data["Grupo_EFA_n"] = pd.to_numeric(
            data["Grupo_EFA_n"], errors="coerce"
        ).fillna(0).astype(int)
        data["n"] = pd.to_numeric(
            data["n"], errors="coerce"
        ).fillna(0).astype(int)
        data["Valor"] = pd.to_numeric(
            data["Valor"], errors="coerce"
        )
        data["Escala_max"] = pd.to_numeric(
            data["Escala_max"], errors="coerce"
        )
        data["Extra"] = pd.to_numeric(
            data["Extra"], errors="coerce"
        )

    return data


@st.cache_data(show_spinner=False)
def process_raw(raw: pd.DataFrame):
    attempts = build_attempt_table(raw)
    attempts["Prueba"] = attempts["QuizName"].apply(normalize_area)
    attempts = attempts[
        (attempts["Año"] == YEAR)
        & (attempts["Prueba"].isin(TESTS))
        & (attempts["Sede"].astype(str).isin(SITES))
    ].copy()

    dimensions = build_dimension_table(raw)
    if not dimensions.empty:
        dimensions = dimensions[
            (dimensions["Año"] == YEAR)
            & (dimensions["Prueba"].isin(TESTS))
            & (dimensions["Sede"].astype(str).isin(SITES))
        ].copy()

    age_attempts = attempts[
        attempts["Sede"].astype(str).isin(ANTIGUEDAD_SITES)
        & (
            pd.to_numeric(
                attempts["AntiguedadBS"],
                errors="coerce",
            )
            <= 1
        )
    ].copy()

    if (
        not dimensions.empty
        and "AntiguedadBS" in dimensions.columns
    ):
        age_dimensions = dimensions[
            dimensions["Sede"].astype(str).isin(ANTIGUEDAD_SITES)
            & (
                pd.to_numeric(
                    dimensions["AntiguedadBS"],
                    errors="coerce",
                )
                <= 1
            )
        ].copy()
    else:
        raw_age = raw[
            raw["Sede"].astype(str).isin(ANTIGUEDAD_SITES)
            & (
                pd.to_numeric(
                    raw["AntiguedadBS"],
                    errors="coerce",
                )
                <= 1
            )
        ].copy()
        age_dimensions = build_dimension_table(raw_age)
        if not age_dimensions.empty:
            age_dimensions = age_dimensions[
                (age_dimensions["Año"] == YEAR)
                & (age_dimensions["Prueba"].isin(TESTS))
            ].copy()

    return attempts, dimensions, age_attempts, age_dimensions


@st.cache_data(show_spinner=False)
def read_default_data():
    bundle_paths = [
        Path(f"data/dashboard_bundle_20261019_part{i:02d}.b64")
        for i in range(14)
    ]
    if all(path.exists() for path in bundle_paths):
        payload = "".join(
            path.read_text(encoding="utf-8").strip()
            for path in bundle_paths
        )
        bundle = json.loads(
            gzip.decompress(
                base64.b64decode(payload)
            ).decode("utf-8")
        )

        temporal = pd.read_csv(
            io.StringIO(bundle["general_temporal"])
        )
        levels = pd.read_csv(
            io.StringIO(bundle["general_levels"])
        )
        dimensions = pd.read_csv(
            io.StringIO(bundle["general_dimensions"])
        )
        top10_origin = pd.read_csv(
            io.StringIO(bundle["general_top10"])
        )
        age_temporal = pd.read_csv(
            io.StringIO(bundle["age_temporal"])
        )
        age_levels = pd.read_csv(
            io.StringIO(bundle["age_levels"])
        )
        age_dimensions = pd.read_csv(
            io.StringIO(bundle["age_dimensions"])
        )
        age_top10 = pd.read_csv(
            io.StringIO(bundle["age_top10"])
        )

        if "Prueba" not in temporal.columns:
            temporal["Prueba"] = temporal["QuizName"].apply(
                normalize_area
            )

        for df in [
            levels,
            top10_origin,
            age_levels,
            age_top10,
        ]:
            if not df.empty and "Nivel" in df.columns:
                df["Nivel"] = df["Nivel"].replace(
                    LEGACY_LEVEL_MAP
                )

        if (
            not dimensions.empty
            and "competencia" in dimensions.columns
        ):
            dimensions = dimensions.rename(
                columns={"competencia": "Dimension"}
            )
        if (
            not age_dimensions.empty
            and "competencia" in age_dimensions.columns
        ):
            age_dimensions = age_dimensions.rename(
                columns={"competencia": "Dimension"}
            )

        if "Año" in temporal.columns:
            temporal = temporal[
                temporal["Año"] == YEAR
            ].copy()
        if "Año" in age_temporal.columns:
            age_temporal = age_temporal[
                age_temporal["Año"] == YEAR
            ].copy()

        temporal = temporal[
            temporal["Prueba"].isin(TESTS)
        ].copy()
        age_temporal = age_temporal[
            age_temporal["Prueba"].isin(TESTS)
        ].copy()

        # NIZA se excluye completamente del análisis general.
        temporal = temporal[
            temporal["Sede"].astype(str).isin(SITES)
        ].copy()
        if not levels.empty and "Sede" in levels.columns:
            levels = levels[
                levels["Sede"].astype(str).isin(SITES)
            ].copy()
        if not dimensions.empty and "Sede" in dimensions.columns:
            dimensions = dimensions[
                dimensions["Sede"].astype(str).isin(SITES)
            ].copy()
        if not top10_origin.empty and "Sede" in top10_origin.columns:
            top10_origin = top10_origin[
                top10_origin["Sede"].astype(str).isin(SITES)
            ].copy()

        return (
            temporal,
            levels,
            dimensions,
            top10_origin,
            age_temporal,
            age_levels,
            age_dimensions,
            age_top10,
        )

    temporal_path = Path("data/default_temporal_summary.csv")
    if not temporal_path.exists():
        return tuple(pd.DataFrame() for _ in range(8))

    temporal = pd.read_csv(temporal_path)
    temporal["Prueba"] = temporal["QuizName"].apply(normalize_area)
    if "Año" in temporal.columns:
        temporal = temporal[temporal["Año"] == YEAR]
    temporal = temporal[temporal["Prueba"].isin(TESTS)].copy()

    level_files = sorted(Path("data").glob("default_levels_2026_part*.csv"))
    levels = (
        pd.concat(
            [pd.read_csv(path) for path in level_files],
            ignore_index=True,
        )
        if level_files
        else pd.DataFrame()
    )
    if not levels.empty and "Nivel" in levels.columns:
        levels["Nivel"] = levels["Nivel"].replace(LEGACY_LEVEL_MAP)

    dimension_files = sorted(
        Path("data").glob("default_dimensions_2026_part*.csv")
    )
    dimensions = (
        pd.concat(
            [pd.read_csv(path) for path in dimension_files],
            ignore_index=True,
        )
        if dimension_files
        else pd.DataFrame()
    )
    if not dimensions.empty and "competencia" in dimensions.columns:
        dimensions = dimensions.rename(
            columns={"competencia": "Dimension"}
        )

    top10_grade_files = sorted(
        Path("data").glob("top10_origin_by_site_grade_*.csv")
    )
    top10_origin = (
        pd.concat(
            [pd.read_csv(path) for path in top10_grade_files],
            ignore_index=True,
        )
        if top10_grade_files
        else pd.DataFrame()
    )
    if not top10_origin.empty and "Nivel" in top10_origin.columns:
        top10_origin["Nivel"] = top10_origin["Nivel"].replace(
            LEGACY_LEVEL_MAP
        )

    age_temporal_path = Path("data/antiguedad_temporal_2026.csv")
    age_temporal = (
        pd.read_csv(age_temporal_path)
        if age_temporal_path.exists()
        else pd.DataFrame()
    )
    if not age_temporal.empty:
        age_temporal = age_temporal[
            age_temporal["Prueba"].isin(TESTS)
        ].copy()

    age_levels_path = Path("data/antiguedad_levels_2026.csv")
    age_levels = (
        pd.read_csv(age_levels_path)
        if age_levels_path.exists()
        else pd.DataFrame()
    )
    if not age_levels.empty and "Nivel" in age_levels.columns:
        age_levels["Nivel"] = age_levels["Nivel"].replace(
            LEGACY_LEVEL_MAP
        )

    age_dimensions_path = Path("data/antiguedad_dimensions_2026.csv")
    age_dimensions = (
        pd.read_csv(age_dimensions_path)
        if age_dimensions_path.exists()
        else pd.DataFrame()
    )

    age_top10_path = Path("data/antiguedad_top10_2026.csv")
    age_top10 = (
        pd.read_csv(age_top10_path)
        if age_top10_path.exists()
        else pd.DataFrame()
    )
    if not age_top10.empty and "Nivel" in age_top10.columns:
        age_top10["Nivel"] = age_top10["Nivel"].replace(
            LEGACY_LEVEL_MAP
        )

    # NIZA se excluye completamente del análisis general.
    temporal = temporal[
        temporal["Sede"].astype(str).isin(SITES)
    ].copy()
    if not levels.empty and "Sede" in levels.columns:
        levels = levels[
            levels["Sede"].astype(str).isin(SITES)
        ].copy()
    if not dimensions.empty and "Sede" in dimensions.columns:
        dimensions = dimensions[
            dimensions["Sede"].astype(str).isin(SITES)
        ].copy()
    if not top10_origin.empty and "Sede" in top10_origin.columns:
        top10_origin = top10_origin[
            top10_origin["Sede"].astype(str).isin(SITES)
        ].copy()

    return (
        temporal,
        levels,
        dimensions,
        top10_origin,
        age_temporal,
        age_levels,
        age_dimensions,
        age_top10,
    )


def grade_label(value) -> str:
    try:
        return f"{int(float(value))}°"
    except Exception:
        return str(value)


def style_clean_axes(fig, grades=None, hide_y=True):
    if grades is not None:
        fig.update_xaxes(
            title=None,
            tickmode="array",
            tickvals=grades,
            ticktext=[grade_label(g) for g in grades],
            showgrid=False,
            zeroline=False,
            ticks="",
        )
    else:
        fig.update_xaxes(
            title=None,
            showgrid=False,
            zeroline=False,
            ticks="",
        )

    fig.update_yaxes(
        title=None,
        showgrid=False,
        zeroline=False,
        showticklabels=not hide_y,
        ticks="",
        rangemode="tozero",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=70, b=25),
    )
    return fig


def build_student_level(data: pd.DataFrame) -> pd.DataFrame:
    student = (
        data.groupby(
            ["IdentiEstudiante", "Sede", "Grado_num"],
            as_index=False,
        )
        .agg(
            Porcentaje=("Porcentaje_acierto", "mean"),
            colegio_de_origen=("colegio_de_origen", "first"),
            AntiguedadBS=("AntiguedadBS", "first"),
        )
    )
    student["AntiguedadBS"] = pd.to_numeric(
        student["AntiguedadBS"],
        errors="coerce",
    )
    student["Nivel"] = np.select(
        [
            student["Porcentaje"] <= 25,
            student["Porcentaje"] <= 50,
            student["Porcentaje"] <= 75,
        ],
        [
            "Nivel 1",
            "Nivel 2",
            "Nivel 3",
        ],
        default="Nivel 4",
    )
    student["Menor_igual_50"] = student["Porcentaje"] <= 50
    return student


def render_threshold_note():
    st.markdown(
        """
        <div class="threshold-box">
        <strong>Lectura del corte del 50%:</strong>
        estar en <strong>≤50% de aciertos</strong> significa pertenecer a
        <strong>Nivel 1 (0–25%)</strong> o
        <strong>Nivel 2 (&gt;25–50%)</strong>.
        Los estudiantes con <strong>&gt;50%</strong> se ubican en
        <strong>Nivel 3</strong> o <strong>Nivel 4</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_main_line(
    d: pd.DataFrame,
    aggregated: bool,
    level_summary: pd.DataFrame | None = None,
    test_name: str | None = None,
):
    if aggregated:
        grouped = (
            d.groupby(["Sede", "Grado_num"], as_index=False)
            .agg(
                Estudiantes=("Estudiantes", "sum"),
                Menor_igual_50=("Debajo_igual_50", "sum"),
                Mayor_50=("Encima_50", "sum"),
            )
        )

        if (
            level_summary is not None
            and not level_summary.empty
            and test_name is not None
        ):
            level_base = level_summary[
                level_summary["Prueba"].astype(str) == test_name
            ].copy()
            level_base = level_base[
                level_base["Sede"].isin(d["Sede"].astype(str).unique())
                & level_base["Grado_num"].isin(d["Grado_num"].unique())
            ]
            level_counts = (
                level_base.groupby(
                    ["Sede", "Grado_num", "Nivel"],
                    as_index=False,
                )["Estudiantes"]
                .sum()
            )
        else:
            level_counts = pd.DataFrame()
    else:
        student = build_student_level(d)
        grouped = (
            student.groupby(["Sede", "Grado_num"], as_index=False)
            .agg(
                Estudiantes=("IdentiEstudiante", "nunique"),
                Menor_igual_50=("Menor_igual_50", "sum"),
            )
        )
        grouped["Mayor_50"] = (
            grouped["Estudiantes"] - grouped["Menor_igual_50"]
        )
        level_counts = (
            student.groupby(
                ["Sede", "Grado_num", "Nivel"],
                as_index=False,
            )
            .agg(Estudiantes=("IdentiEstudiante", "nunique"))
        )

    if grouped.empty:
        st.info("No hay datos para los filtros seleccionados.")
        return

    grouped["Pct_50_o_menos"] = np.where(
        grouped["Estudiantes"] > 0,
        grouped["Menor_igual_50"] / grouped["Estudiantes"] * 100,
        np.nan,
    )

    if not level_counts.empty:
        level_pivot = (
            level_counts.pivot_table(
                index=["Sede", "Grado_num"],
                columns="Nivel",
                values="Estudiantes",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        for level in LEVEL_ORDER:
            if level not in level_pivot.columns:
                level_pivot[level] = 0

        level_pivot["Total_niveles"] = level_pivot[LEVEL_ORDER].sum(axis=1)
        level_pivot["Pct_Nivel_1"] = np.where(
            level_pivot["Total_niveles"] > 0,
            level_pivot["Nivel 1"] / level_pivot["Total_niveles"] * 100,
            0,
        )
        level_pivot["Pct_Nivel_2"] = np.where(
            level_pivot["Total_niveles"] > 0,
            level_pivot["Nivel 2"] / level_pivot["Total_niveles"] * 100,
            0,
        )
        level_pivot["Pct_Nivel1_2"] = (
            level_pivot["Pct_Nivel_1"] + level_pivot["Pct_Nivel_2"]
        )

        grouped = grouped.merge(
            level_pivot[
                [
                    "Sede",
                    "Grado_num",
                    "Pct_Nivel_1",
                    "Pct_Nivel_2",
                    "Pct_Nivel1_2",
                ]
            ],
            on=["Sede", "Grado_num"],
            how="left",
        )
    else:
        grouped["Pct_Nivel_1"] = np.nan
        grouped["Pct_Nivel_2"] = np.nan
        grouped["Pct_Nivel1_2"] = grouped["Pct_50_o_menos"]

    grouped["Etiqueta_Pct_50"] = grouped["Pct_50_o_menos"].map(
        lambda x: "" if pd.isna(x) else f"{x:.0f}%"
    )

    total_low = int(grouped["Menor_igual_50"].sum())
    total_high = int(grouped["Mayor_50"].sum())
    total = total_low + total_high

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Nivel 1 + Nivel 2",
        f"{total_low:,}".replace(",", "."),
        help="Estudiantes con 50% o menos de aciertos.",
    )
    m2.metric(
        "Nivel 3 + Nivel 4",
        f"{total_high:,}".replace(",", "."),
        help="Estudiantes con más de 50% de aciertos.",
    )
    m3.metric(
        "% en Nivel 1 + Nivel 2",
        f"{(total_low / total * 100):.1f}%" if total else "—",
    )

    grades = sorted(
        [int(g) for g in grouped["Grado_num"].dropna().unique()]
    )

    fig = px.line(
        grouped.sort_values(["Sede", "Grado_num"]),
        x="Grado_num",
        y="Pct_50_o_menos",
        color="Sede",
        markers=True,
        text="Etiqueta_Pct_50",
        category_orders={"Sede": SITES},
        custom_data=[
            "Pct_Nivel_1",
            "Pct_Nivel_2",
            "Pct_Nivel1_2",
        ],
        labels={
            "Sede": "Sede",
            "Pct_50_o_menos": "% en Nivel 1 + Nivel 2",
        },
        title="% de estudiantes en Nivel 1 + Nivel 2 por grado",
    )
    fig = style_clean_axes(fig, grades=grades, hide_y=False)
    fig.update_yaxes(
        range=[0, 100],
        showticklabels=True,
        ticksuffix="%",
        title="% en Nivel 1 + Nivel 2",
    )
    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend_title_text="Sede",
    )
    fig.update_traces(
        textposition="top center",
        cliponaxis=False,
        line=dict(width=2.5),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Nivel 1: %{customdata[0]:.1f}%<br>"
            "Nivel 2: %{customdata[1]:.1f}%<br>"
            "Nivel 1 + Nivel 2: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def level_counts_by_site_from_raw(data: pd.DataFrame) -> pd.DataFrame:
    student = build_student_level(data)
    return (
        student.groupby(["Sede", "Nivel"], as_index=False)
        .agg(Estudiantes=("IdentiEstudiante", "nunique"))
    )


def render_levels_by_site_vs_network(
    site_data: pd.DataFrame,
    network_data: pd.DataFrame,
    aggregated: bool,
    level_summary: pd.DataFrame | None,
    test_name: str,
    selected_sites: list[str],
    selected_grades: list[int],
):
    st.subheader("Vista 1 · Distribución de los 4 niveles por colegio/sede")
    st.caption(
        "Cada fila representa una sede/colegio. La fila RED muestra el "
        "comportamiento global de las sedes incluidas en esta vista para la "
        "misma prueba y los mismos grados seleccionados."
    )

    if aggregated:
        if level_summary is None or level_summary.empty:
            st.info(
                "La base precargada no contiene el desglose de los cuatro niveles."
            )
            return

        base = level_summary[
            level_summary["Prueba"].astype(str) == test_name
        ].copy()

        if selected_grades:
            base = base[base["Grado_num"].isin(selected_grades)]

        site_counts = (
            base[base["Sede"].isin(selected_sites)]
            .groupby(["Sede", "Nivel"], as_index=False)["Estudiantes"]
            .sum()
        )

        network_counts = (
            base.groupby("Nivel", as_index=False)["Estudiantes"]
            .sum()
            .assign(Sede="RED")
        )
    else:
        site_counts = level_counts_by_site_from_raw(site_data)
        network_counts = (
            level_counts_by_site_from_raw(network_data)
            .groupby("Nivel", as_index=False)["Estudiantes"]
            .sum()
            .assign(Sede="RED")
        )

    counts = pd.concat([site_counts, network_counts], ignore_index=True)
    if counts.empty:
        st.info("No hay datos de niveles para los filtros seleccionados.")
        return

    entities = [
        site for site in SITES
        if site in selected_sites
        and site in counts["Sede"].astype(str).unique()
    ] + ["RED"]

    full = pd.MultiIndex.from_product(
        [entities, LEVEL_ORDER],
        names=["Sede", "Nivel"],
    ).to_frame(index=False)

    counts = full.merge(
        counts,
        on=["Sede", "Nivel"],
        how="left",
    )
    counts["Estudiantes"] = counts["Estudiantes"].fillna(0).astype(int)

    totals = counts.groupby("Sede")["Estudiantes"].transform("sum")
    counts["Porcentaje"] = np.where(
        totals > 0,
        counts["Estudiantes"] / totals * 100,
        0,
    )
    counts["Etiqueta"] = counts.apply(
        lambda r: (
            f"{r['Porcentaje']:.0f}%"
            if r["Porcentaje"] >= 6 and r["Estudiantes"] > 0
            else ""
        ),
        axis=1,
    )

    fig = px.bar(
        counts,
        x="Porcentaje",
        y="Sede",
        color="Nivel",
        orientation="h",
        barmode="stack",
        category_orders={
            "Sede": entities,
            "Nivel": LEVEL_ORDER,
        },
        color_discrete_map=LEVEL_COLORS,
        text="Etiqueta",
        hover_data={
            "Estudiantes": True,
            "Porcentaje": ":.1f",
            "Etiqueta": False,
        },
        labels={
            "Porcentaje": "",
            "Sede": "",
            "Nivel": "Nivel",
        },
        title="Composición de niveles: cada colegio/sede frente a la RED",
    )
    fig.update_xaxes(
        range=[0, 100],
        showticklabels=False,
        showgrid=False,
        ticks="",
        title=None,
    )
    fig.update_yaxes(
        showgrid=False,
        ticks="",
        title=None,
        categoryorder="array",
        categoryarray=list(reversed(entities)),
    )
    fig.update_layout(
        height=max(430, 62 * len(entities) + 120),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=70, b=20),
        legend_title_text="Nivel",
    )
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

    network_pct = (
        counts[counts["Sede"] == "RED"]
        .set_index("Nivel")["Porcentaje"]
        .reindex(LEVEL_ORDER)
    )

    site_pct = counts[counts["Sede"] != "RED"].copy()
    site_pct["Diferencia_vs_RED"] = site_pct.apply(
        lambda r: r["Porcentaje"] - network_pct.loc[r["Nivel"]],
        axis=1,
    )

    if not site_pct.empty:
        delta = (
            site_pct.pivot(
                index="Sede",
                columns="Nivel",
                values="Diferencia_vs_RED",
            )
            .reindex(
                index=[site for site in entities if site != "RED"],
                columns=LEVEL_ORDER,
            )
        )

        labels = delta.map(
            lambda x: "" if pd.isna(x) else f"{x:+.1f} pp"
        )

        max_abs = np.nanmax(np.abs(delta.values))
        max_abs = max(
            10,
            min(40, max_abs if np.isfinite(max_abs) else 10),
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=delta.values,
                x=delta.columns,
                y=delta.index,
                zmid=0,
                zmin=-max_abs,
                zmax=max_abs,
                colorscale=[
                    [0.0, "#C93C3C"],
                    [0.5, "#F7F7F7"],
                    [1.0, "#2E8B57"],
                ],
                text=labels.values,
                texttemplate="%{text}",
                hovertemplate=(
                    "Sede: %{y}<br>Nivel: %{x}<br>"
                    "Diferencia vs RED: %{z:+.1f} pp<extra></extra>"
                ),
                colorbar=dict(title="pp vs RED"),
            )
        )
        fig.update_layout(
            title=(
                "Contraste por colegio/sede frente a la RED · "
                "rojo = por debajo · verde = por encima"
            ),
            height=max(360, 55 * len(delta.index) + 140),
            margin=dict(l=18, r=18, t=65, b=25),
        )
        st.plotly_chart(fig, use_container_width=True)


def level_counts_by_grade_from_raw(data: pd.DataFrame) -> pd.DataFrame:
    student = build_student_level(data)
    return (
        student.groupby(["Grado_num", "Nivel"], as_index=False)
        .agg(Estudiantes=("IdentiEstudiante", "nunique"))
    )


def render_levels_vs_network(
    site_data: pd.DataFrame,
    network_data: pd.DataFrame,
    aggregated: bool,
    level_summary: pd.DataFrame | None,
    test_name: str,
    selected_sites: list[str],
    selected_grades: list[int],
):
    st.subheader("Vista 2 · Distribución de los 4 niveles por grado")
    st.caption(
        "Cada fila representa un grado. La fila RED muestra el comportamiento "
        "global de las sedes incluidas en esta vista para la misma prueba y los mismos grados seleccionados."
    )

    if aggregated:
        if level_summary is None or level_summary.empty:
            st.info(
                "La base pública precargada aún no contiene el desglose de los cuatro niveles. "
                "Carga el Excel completo para habilitar esta comparación."
            )
            return

        base = level_summary[
            level_summary["Prueba"].astype(str) == test_name
        ].copy()

        if selected_grades:
            base = base[base["Grado_num"].isin(selected_grades)]

        # Distribución por grado usando únicamente las sedes seleccionadas.
        grade_counts = (
            base[base["Sede"].isin(selected_sites)]
            .groupby(["Grado_num", "Nivel"], as_index=False)["Estudiantes"]
            .sum()
        )

        # RED = seis sedes, independientemente del filtro de sede.
        network_counts = (
            base.groupby("Nivel", as_index=False)["Estudiantes"]
            .sum()
            .assign(Grado_num="RED")
        )
    else:
        grade_counts = level_counts_by_grade_from_raw(site_data)
        network_counts = (
            level_counts_by_grade_from_raw(network_data)
            .groupby("Nivel", as_index=False)["Estudiantes"]
            .sum()
            .assign(Grado_num="RED")
        )

    if grade_counts.empty and network_counts.empty:
        st.info("No hay datos de niveles para los filtros seleccionados.")
        return

    grade_counts = grade_counts.copy()
    grade_counts["Grado"] = grade_counts["Grado_num"].apply(grade_label)

    network_counts = network_counts.copy()
    network_counts["Grado"] = "RED"

    counts = pd.concat(
        [
            grade_counts[["Grado", "Nivel", "Estudiantes"]],
            network_counts[["Grado", "Nivel", "Estudiantes"]],
        ],
        ignore_index=True,
    )

    grade_entities = [
        grade_label(g)
        for g in sorted(
            grade_counts["Grado_num"].dropna().astype(int).unique()
        )
    ]
    entities = grade_entities + ["RED"]

    full = pd.MultiIndex.from_product(
        [entities, LEVEL_ORDER],
        names=["Grado", "Nivel"],
    ).to_frame(index=False)

    counts = full.merge(
        counts,
        on=["Grado", "Nivel"],
        how="left",
    )
    counts["Estudiantes"] = counts["Estudiantes"].fillna(0).astype(int)

    totals = counts.groupby("Grado")["Estudiantes"].transform("sum")
    counts["Porcentaje"] = np.where(
        totals > 0,
        counts["Estudiantes"] / totals * 100,
        0,
    )
    counts["Etiqueta"] = counts.apply(
        lambda r: (
            f"{r['Porcentaje']:.0f}%"
            if r["Porcentaje"] >= 6 and r["Estudiantes"] > 0
            else ""
        ),
        axis=1,
    )

    fig = px.bar(
        counts,
        x="Porcentaje",
        y="Grado",
        color="Nivel",
        orientation="h",
        barmode="stack",
        category_orders={
            "Grado": entities,
            "Nivel": LEVEL_ORDER,
        },
        color_discrete_map=LEVEL_COLORS,
        text="Etiqueta",
        hover_data={
            "Estudiantes": True,
            "Porcentaje": ":.1f",
            "Etiqueta": False,
        },
        labels={
            "Porcentaje": "",
            "Grado": "",
            "Nivel": "Nivel",
        },
        title="Composición de niveles: cada grado frente a la RED",
    )
    fig.update_xaxes(
        range=[0, 100],
        showticklabels=False,
        showgrid=False,
        ticks="",
        title=None,
    )
    fig.update_yaxes(
        showgrid=False,
        ticks="",
        title=None,
        categoryorder="array",
        categoryarray=list(reversed(entities)),
    )
    fig.update_layout(
        height=max(430, 62 * len(entities) + 120),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=70, b=20),
        legend_title_text="Nivel",
    )
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

    network_pct = (
        counts[counts["Grado"] == "RED"]
        .set_index("Nivel")["Porcentaje"]
        .reindex(LEVEL_ORDER)
    )

    grade_pct = counts[counts["Grado"] != "RED"].copy()
    grade_pct["Diferencia_vs_RED"] = grade_pct.apply(
        lambda r: r["Porcentaje"] - network_pct.loc[r["Nivel"]],
        axis=1,
    )

    if not grade_pct.empty:
        delta = (
            grade_pct.pivot(
                index="Grado",
                columns="Nivel",
                values="Diferencia_vs_RED",
            )
            .reindex(
                index=grade_entities,
                columns=LEVEL_ORDER,
            )
        )

        labels = delta.map(
            lambda x: "" if pd.isna(x) else f"{x:+.1f} pp"
        )

        max_abs = np.nanmax(np.abs(delta.values))
        max_abs = max(
            10,
            min(40, max_abs if np.isfinite(max_abs) else 10),
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=delta.values,
                x=delta.columns,
                y=delta.index,
                zmid=0,
                zmin=-max_abs,
                zmax=max_abs,
                colorscale=[
                    [0.0, "#C93C3C"],
                    [0.5, "#F7F7F7"],
                    [1.0, "#2E8B57"],
                ],
                text=labels.values,
                texttemplate="%{text}",
                hovertemplate=(
                    "Grado: %{y}<br>Nivel: %{x}<br>"
                    "Diferencia vs RED: %{z:+.1f} pp<extra></extra>"
                ),
                colorbar=dict(title="pp vs RED"),
            )
        )
        fig.update_layout(
            title="Contraste por grado frente a la RED · rojo = por debajo · verde = por encima",
            height=max(360, 55 * len(delta.index) + 140),
            margin=dict(l=18, r=18, t=65, b=25),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_origin_patterns(d: pd.DataFrame):
    if (
        "colegio_de_origen" not in d.columns
        or d["colegio_de_origen"].dropna().empty
    ):
        return

    origin = d[
        d["colegio_de_origen"].notna()
        & (d["colegio_de_origen"].astype(str).str.strip() != "")
    ].copy()
    if origin.empty:
        return

    student = (
        origin.groupby(
            ["IdentiEstudiante", "colegio_de_origen", "Grado_num"],
            as_index=False,
        )
        .agg(Porcentaje=("Porcentaje_acierto", "mean"))
    )
    student["Bajo_50"] = student["Porcentaje"] <= 50

    school_volume = (
        student.groupby("colegio_de_origen", as_index=False)
        .agg(Estudiantes=("IdentiEstudiante", "nunique"))
        .sort_values("Estudiantes", ascending=False)
    )

    top = school_volume[
        school_volume["Estudiantes"] >= 5
    ].head(8)
    if top.empty:
        return

    sd = student[
        student["colegio_de_origen"].isin(top["colegio_de_origen"])
    ]
    trend = (
        sd.groupby(["colegio_de_origen", "Grado_num"], as_index=False)
        .agg(
            Estudiantes=("IdentiEstudiante", "nunique"),
            Pct_50=("Bajo_50", "mean"),
        )
    )
    trend["Pct_50"] *= 100

    grades = sorted(
        [int(g) for g in trend["Grado_num"].dropna().unique()]
    )
    fig = px.line(
        trend,
        x="Grado_num",
        y="Pct_50",
        color="colegio_de_origen",
        markers=True,
        text=trend["Pct_50"].map(lambda x: f"{x:.0f}%"),
        labels={
            "colegio_de_origen": "Colegio de origen",
            "Pct_50": "% ≤50",
        },
        title="Patrón de estudiantes ≤50% por colegio de origen",
    )
    fig = style_clean_axes(fig, grades=grades, hide_y=True)
    fig.update_yaxes(range=[0, 100], showticklabels=False)
    fig.update_layout(
        height=540,
        legend_title_text="Colegio de origen",
    )
    fig.update_traces(
        textposition="top center",
        cliponaxis=False,
        line=dict(width=2.2),
        marker=dict(size=7),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top10_origin_by_level(
    data: pd.DataFrame,
    test_name: str,
    aggregated: bool,
    top10_public: pd.DataFrame | None,
    view_key: str,
):
    st.divider()
    st.subheader("Colegios de origen por nivel de desempeño")
    st.caption(
        "Filtra por sede de llegada y por grado. Después se muestran los "
        "10 colegios de origen con mayor número de estudiantes en cada nivel."
    )

    if aggregated:
        if top10_public is None or top10_public.empty:
            st.info("No hay un resumen público de colegios de origen disponible.")
            return

        base = top10_public[
            top10_public["Prueba"].astype(str) == test_name
        ].copy()

        if base.empty:
            st.info("No hay información de colegios de origen para esta prueba.")
            return

        available_sites = [
            site for site in SITES
            if site in base["Sede"].dropna().astype(str).unique()
        ]

        c1, c2 = st.columns([1, 1.2])
        with c1:
            selected_origin_sites = st.multiselect(
                "Sede",
                available_sites,
                default=available_sites,
                key=f"origin_sites_{view_key}_{test_name}",
                help="Sede de llegada del estudiante.",
            )

        filtered = base.copy()
        if selected_origin_sites:
            filtered = filtered[
                filtered["Sede"].astype(str).isin(selected_origin_sites)
            ]

        grade_options = sorted(
            [int(g) for g in filtered["Grado_num"].dropna().unique()]
        )
        with c2:
            selected_origin_grades = st.multiselect(
                "Grado",
                grade_options,
                default=grade_options,
                format_func=lambda g: f"{g}°",
                key=f"origin_grades_{view_key}_{test_name}",
            )

        if selected_origin_grades:
            filtered = filtered[
                filtered["Grado_num"].isin(selected_origin_grades)
            ]

        if filtered.empty:
            st.warning("Los filtros de sede y grado no dejan datos para analizar.")
            return

        ranking = (
            filtered.groupby(
                ["Nivel", "colegio_de_origen"],
                as_index=False,
            )["Estudiantes"]
            .sum()
        )

        m1, m2 = st.columns(2)
        m1.metric(
            "Estudiantes representados en el Top 10",
            f"{int(ranking['Estudiantes'].sum()):,}".replace(",", "."),
        )
        m2.metric(
            "Colegios de origen representados",
            f"{ranking['colegio_de_origen'].nunique():,}".replace(",", "."),
        )

    else:
        if (
            "colegio_de_origen" not in data.columns
            or data["colegio_de_origen"].dropna().empty
        ):
            st.info("No hay información de colegio de origen en los filtros actuales.")
            return

        student = build_student_level(data)
        student = student[
            student["colegio_de_origen"].notna()
            & (student["colegio_de_origen"].astype(str).str.strip() != "")
        ].copy()

        if student.empty:
            st.info("No hay información de colegio de origen en los filtros actuales.")
            return

        available_sites = [
            site for site in SITES
            if site in student["Sede"].dropna().astype(str).unique()
        ]

        c1, c2 = st.columns([1, 1.2])
        with c1:
            selected_origin_sites = st.multiselect(
                "Sede",
                available_sites,
                default=available_sites,
                key=f"origin_sites_{view_key}_{test_name}",
                help="Sede de llegada del estudiante.",
            )

        filtered_students = student.copy()
        if selected_origin_sites:
            filtered_students = filtered_students[
                filtered_students["Sede"].astype(str).isin(selected_origin_sites)
            ]

        grade_options = sorted(
            [int(g) for g in filtered_students["Grado_num"].dropna().unique()]
        )
        with c2:
            selected_origin_grades = st.multiselect(
                "Grado",
                grade_options,
                default=grade_options,
                format_func=lambda g: f"{g}°",
                key=f"origin_grades_{view_key}_{test_name}",
            )

        if selected_origin_grades:
            filtered_students = filtered_students[
                filtered_students["Grado_num"].isin(selected_origin_grades)
            ]

        if filtered_students.empty:
            st.warning("Los filtros de sede y grado no dejan estudiantes para analizar.")
            return

        total_students = filtered_students["IdentiEstudiante"].nunique()
        total_origin_schools = filtered_students["colegio_de_origen"].nunique()

        m1, m2 = st.columns(2)
        m1.metric(
            "Estudiantes en la selección",
            f"{total_students:,}".replace(",", "."),
        )
        m2.metric(
            "Colegios de origen representados",
            f"{total_origin_schools:,}".replace(",", "."),
        )

        ranking = (
            filtered_students.groupby(
                ["Nivel", "colegio_de_origen"],
                as_index=False,
            )
            .agg(Estudiantes=("IdentiEstudiante", "nunique"))
        )

    st.markdown(
        "#### Top 10 colegios de origen con mayor número de estudiantes en cada nivel"
    )

    ranking = ranking.sort_values(
        ["Nivel", "Estudiantes", "colegio_de_origen"],
        ascending=[True, False, True],
    )
    ranking["Ranking"] = (
        ranking.groupby("Nivel")["Estudiantes"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    ranking = ranking[ranking["Ranking"] <= 10].copy()

    rows = [
        ("Nivel 1", "Nivel 2"),
        ("Nivel 3", "Nivel 4"),
    ]

    for left_level, right_level in rows:
        c1, c2 = st.columns(2)

        for col, level in [(c1, left_level), (c2, right_level)]:
            with col:
                d = ranking[
                    ranking["Nivel"].astype(str) == level
                ].copy()

                if d.empty:
                    st.info(f"Sin datos para {level}.")
                    continue

                d = d.sort_values(
                    ["Estudiantes", "colegio_de_origen"],
                    ascending=[True, False],
                )

                fig = px.bar(
                    d,
                    x="Estudiantes",
                    y="colegio_de_origen",
                    orientation="h",
                    text="Estudiantes",
                    title=level,
                    labels={
                        "Estudiantes": "",
                        "colegio_de_origen": "",
                    },
                )
                fig.update_traces(
                    marker_color=LEVEL_COLORS[level],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Estudiantes: %{x}"
                        "<extra></extra>"
                    ),
                )
                fig.update_xaxes(
                    showgrid=False,
                    showticklabels=False,
                    ticks="",
                    title=None,
                    rangemode="tozero",
                )
                fig.update_yaxes(
                    showgrid=False,
                    ticks="",
                    title=None,
                )
                fig.update_layout(
                    height=430,
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=12, r=35, t=55, b=15),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)


def render_dimensions(
    dimensions: pd.DataFrame,
    test_name: str,
    selected_sites: list[str],
    selected_grades: list[int],
    aggregated: bool,
    view_key: str,
):
    st.divider()
    st.subheader("Dimensiones de evaluación")

    if dimensions is None or dimensions.empty:
        st.info(
            "La base pública precargada aún no contiene el resumen de dimensiones. "
            "Carga el Excel completo para habilitar esta lectura."
        )
        return

    d = dimensions[
        dimensions["Prueba"].astype(str) == test_name
    ].copy()
    if selected_grades:
        d = d[d["Grado_num"].isin(selected_grades)]

    if d.empty:
        st.info("No hay información de dimensiones para los filtros seleccionados.")
        return

    available_sites = [
        s for s in SITES
        if s in d["Sede"].dropna().astype(str).unique()
        and s in selected_sites
    ]
    reference = st.selectbox(
        "Vista de dimensiones",
        ["RED"] + available_sites,
        key=f"dimension_reference_{view_key}_{test_name}",
        help=(
            "RED combina las sedes incluidas en esta vista. Selecciona una sede para ver "
            "el comportamiento de sus dimensiones a través de los grados."
        ),
    )

    if aggregated:
        if reference == "RED":
            plot = (
                d.groupby(
                    ["Grado_num", "Dimension"],
                    as_index=False,
                )
                .apply(
                    lambda g: pd.Series({
                        "Estudiantes": g["Estudiantes"].sum(),
                        "Promedio_dimension": np.average(
                            g["Promedio_dimension"],
                            weights=g["Estudiantes"],
                        ),
                    }),
                    include_groups=False,
                )
            )
        else:
            plot = d[d["Sede"] == reference].copy()
    else:
        if reference == "RED":
            plot = (
                d.groupby(
                    ["Grado_num", "Dimension"],
                    as_index=False,
                )
                .agg(
                    Estudiantes=("IdentiEstudiante", "nunique"),
                    Promedio_dimension=("Porcentaje_dimension", "mean"),
                )
            )
        else:
            plot = (
                d[d["Sede"] == reference]
                .groupby(
                    ["Grado_num", "Dimension"],
                    as_index=False,
                )
                .agg(
                    Estudiantes=("IdentiEstudiante", "nunique"),
                    Promedio_dimension=("Porcentaje_dimension", "mean"),
                )
            )

    if plot.empty:
        st.info("No hay dimensiones para esa selección.")
        return

    plot["Etiqueta"] = plot["Promedio_dimension"].map(
        lambda x: f"{x:.0f}%"
    )
    grades = sorted(
        [int(g) for g in plot["Grado_num"].dropna().unique()]
    )

    fig = px.line(
        plot.sort_values(["Dimension", "Grado_num"]),
        x="Grado_num",
        y="Promedio_dimension",
        color="Dimension",
        markers=True,
        text="Etiqueta",
        hover_data={
            "Grado_num": False,
            "Estudiantes": True,
            "Promedio_dimension": ":.1f",
            "Etiqueta": False,
        },
        labels={
            "Dimension": "Dimensión",
            "Promedio_dimension": "% de acierto",
        },
        title=f"Comportamiento de las dimensiones por grado | {reference}",
    )
    fig = style_clean_axes(fig, grades=grades, hide_y=True)
    fig.update_yaxes(
        range=[0, 100],
        showticklabels=False,
    )
    fig.update_layout(
        height=570,
        legend_title_text="Dimensión",
    )
    fig.update_traces(
        textposition="top center",
        cliponaxis=False,
        line=dict(width=2.2),
        marker=dict(size=7),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "El porcentaje de cada dimensión se calcula sobre los ítems esperados "
        "para esa competencia. Las cifras sobre los puntos muestran el promedio de acierto."
    )


def render_test_page(
    attempts: pd.DataFrame,
    dimensions: pd.DataFrame,
    levels_public: pd.DataFrame,
    top10_public: pd.DataFrame,
    test_name: str,
    aggregated: bool,
    view_key: str,
    cohort_caption: str,
):
    st.markdown(f"## {test_name}")
    st.caption(
        f"Resultados de {test_name} · Año fijo: {YEAR} · {cohort_caption}"
    )
    render_threshold_note()

    test_all = attempts[
        attempts["Prueba"].astype(str) == test_name
    ].copy()
    if test_all.empty:
        st.info(f"No hay datos de {test_name} para esta vista.")
        return

    c1, c2 = st.columns([1.2, 1.4])
    with c1:
        available_sites = [
            site for site in SITES
            if site in test_all["Sede"].dropna().astype(str).unique()
        ]
        selected_sites = st.multiselect(
            "Sedes/colegios",
            available_sites,
            default=available_sites,
            key=f"sites_{view_key}_{test_name}",
        )

    with c2:
        grades = sorted(
            [int(g) for g in test_all["Grado_num"].dropna().unique()]
        )
        selected_grades = st.multiselect(
            "Grados",
            grades,
            default=grades,
            format_func=lambda g: f"{g}°",
            key=f"grades_{view_key}_{test_name}",
        )

    grade_filtered_network = test_all.copy()
    if selected_grades:
        grade_filtered_network = grade_filtered_network[
            grade_filtered_network["Grado_num"].isin(selected_grades)
        ]

    site_filtered = grade_filtered_network.copy()
    if selected_sites:
        site_filtered = site_filtered[
            site_filtered["Sede"].astype(str).isin(selected_sites)
        ]

    if site_filtered.empty:
        st.warning("Los filtros seleccionados no dejan datos.")
        return

    render_main_line(
        site_filtered,
        aggregated=aggregated,
        level_summary=levels_public,
        test_name=test_name,
    )

    st.divider()
    render_levels_by_site_vs_network(
        site_data=site_filtered,
        network_data=grade_filtered_network,
        aggregated=aggregated,
        level_summary=levels_public,
        test_name=test_name,
        selected_sites=selected_sites,
        selected_grades=selected_grades,
    )

    st.divider()
    render_levels_vs_network(
        site_data=site_filtered,
        network_data=grade_filtered_network,
        aggregated=aggregated,
        level_summary=levels_public,
        test_name=test_name,
        selected_sites=selected_sites,
        selected_grades=selected_grades,
    )

    if not aggregated:
        with st.expander(
            "Explorar patrones por colegio de origen",
            expanded=False,
        ):
            render_origin_patterns(site_filtered)

    render_dimensions(
        dimensions=dimensions,
        test_name=test_name,
        selected_sites=selected_sites,
        selected_grades=selected_grades,
        aggregated=aggregated,
        view_key=view_key,
    )

    render_top10_origin_by_level(
        data=site_filtered,
        test_name=test_name,
        aggregated=aggregated,
        top10_public=top10_public,
        view_key=view_key,
    )


def _spearman_from_aggregate(data: pd.DataFrame) -> float:
    if data is None or data.empty:
        return np.nan

    codes = data["Categoria_LATAM"].map(LATAM_SCORE)
    weights = pd.to_numeric(
        data["Estudiantes"], errors="coerce"
    ).fillna(0).astype(int)
    scores = pd.to_numeric(
        data["Pct_EFA"], errors="coerce"
    )

    valid = codes.notna() & scores.notna() & (weights > 0)
    if not valid.any():
        return np.nan

    x = np.repeat(codes[valid].to_numpy(), weights[valid].to_numpy())
    y = np.repeat(scores[valid].to_numpy(), weights[valid].to_numpy())

    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan

    # Spearman = correlación de Pearson entre los rangos.
    # Se calcula explícitamente para no depender de scipy en Streamlit Cloud.
    x_rank = pd.Series(x, dtype="float64").rank(method="average")
    y_rank = pd.Series(y, dtype="float64").rank(method="average")

    if x_rank.std(ddof=0) == 0 or y_rank.std(ddof=0) == 0:
        return np.nan

    return float(np.corrcoef(x_rank.to_numpy(), y_rank.to_numpy())[0, 1])


def _weighted_group_average(group: pd.DataFrame) -> float:
    weights = pd.to_numeric(
        group["Estudiantes"], errors="coerce"
    ).fillna(0)
    values = pd.to_numeric(
        group["Pct_EFA"], errors="coerce"
    )
    valid = values.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def render_latam_efa_area(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    area: str,
):
    st.markdown(f"## {area}")
    st.caption(
        "Promedio de Evaluar para Avanzar (EFA) de los estudiantes "
        "ubicados en cada categoría de logro LATAM."
    )

    d = data[data["Area"].astype(str) == area].copy()
    if d.empty:
        st.info(f"No hay cruces LATAM–EFA disponibles para {area}.")
        return

    srow = summary[
        summary["Area"].astype(str) == area
    ].copy() if summary is not None and not summary.empty else pd.DataFrame()

    available_sites = [
        site for site in SITES
        if site in d["Sede"].dropna().astype(str).unique()
    ]
    grades = sorted(
        [int(g) for g in d["Grado_num"].dropna().unique()]
    )

    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        selected_sites = st.multiselect(
            "Sedes/colegios",
            available_sites,
            default=available_sites,
            key=f"latam_sites_{area}",
        )
    with c2:
        selected_grades = st.multiselect(
            "Grados",
            grades,
            default=grades,
            format_func=lambda g: f"{g}°",
            key=f"latam_grades_{area}",
        )

    filtered = d.copy()
    if selected_sites:
        filtered = filtered[
            filtered["Sede"].astype(str).isin(selected_sites)
        ]
    if selected_grades:
        filtered = filtered[
            filtered["Grado_num"].isin(selected_grades)
        ]

    if filtered.empty:
        st.warning("Los filtros seleccionados no dejan estudiantes emparejados.")
        return

    matched_filtered = int(filtered["Estudiantes"].sum())
    rho = _spearman_from_aggregate(filtered)

    if not srow.empty:
        latam_total = int(srow.iloc[0]["Registros_LATAM"])
        matched_total = int(srow.iloc[0]["Emparejados_EFA_misma_area"])
        coverage = float(srow.iloc[0]["Cobertura_sobre_LATAM_pct"])
    else:
        latam_total = 0
        matched_total = int(d["Estudiantes"].sum())
        coverage = np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros LATAM", f"{latam_total:,}".replace(",", "."))
    m2.metric(
        "Emparejados EFA",
        f"{matched_total:,}".replace(",", "."),
        help="Cruce por correo mediante la base puente y coincidencia de área.",
    )
    m3.metric(
        "Cobertura",
        f"{coverage:.1f}%" if np.isfinite(coverage) else "—",
    )
    m4.metric(
        "ρ Spearman",
        f"{rho:.2f}" if np.isfinite(rho) else "—",
        help=(
            "Asociación ordinal entre la categoría LATAM y el porcentaje "
            "de aciertos en EFA. Se recalcula con los filtros activos."
        ),
    )

    st.caption(
        f"Estudiantes representados en los filtros actuales: "
        f"{matched_filtered:,}. El emparejamiento se realizó por "
        "CORREO → ExternalEmail → OrgDefinedId → IdentiEstudiante. "
        "No se publican correos ni documentos en esta vista."
    )

    # Lectura principal: promedio EFA dentro de cada categoría LATAM.
    category_avg = (
        filtered.groupby(
            ["Categoria_LATAM"],
            as_index=False,
        )
        .apply(
            lambda g: pd.Series({
                "n": int(g["Estudiantes"].sum()),
                "Promedio_EFA": _weighted_group_average(g),
            }),
            include_groups=False,
        )
    )
    category_avg["Categoria_LATAM"] = pd.Categorical(
        category_avg["Categoria_LATAM"],
        categories=LATAM_ORDER,
        ordered=True,
    )
    category_avg = category_avg.sort_values("Categoria_LATAM")
    category_avg["Categoría LATAM"] = (
        category_avg["Categoria_LATAM"].astype(str)
    )
    category_avg["Promedio EFA"] = category_avg["Promedio_EFA"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
    )

    st.markdown("### Promedio EFA dentro de cada categoría LATAM")
    st.dataframe(
        category_avg[
            ["Categoría LATAM", "n", "Promedio EFA"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    fig_avg = px.bar(
        category_avg,
        x="Categoria_LATAM",
        y="Promedio_EFA",
        text=category_avg["Promedio_EFA"].map(
            lambda x: f"{x:.1f}%"
        ),
        category_orders={"Categoria_LATAM": LATAM_ORDER},
        custom_data=["n"],
        labels={
            "Categoria_LATAM": "Categoría LATAM",
            "Promedio_EFA": "Promedio EFA",
        },
        title=(
            "Promedio de Evaluar para Avanzar según categoría LATAM"
        ),
    )
    fig_avg.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
        title="% promedio de aciertos EFA",
    )
    fig_avg.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Promedio EFA: %{y:.1f}%<br>"
            "Estudiantes: %{customdata[0]}<extra></extra>"
        ),
    )
    fig_avg.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # Segunda lectura: el mismo promedio, separado por grado.
    grade_avg = (
        filtered.groupby(
            ["Grado_num", "Categoria_LATAM"],
            as_index=False,
        )
        .apply(
            lambda g: pd.Series({
                "n": int(g["Estudiantes"].sum()),
                "Promedio_EFA": _weighted_group_average(g),
            }),
            include_groups=False,
        )
    )
    grade_avg["Categoria_LATAM"] = pd.Categorical(
        grade_avg["Categoria_LATAM"],
        categories=LATAM_ORDER,
        ordered=True,
    )
    grade_avg = grade_avg.sort_values(
        ["Grado_num", "Categoria_LATAM"]
    )
    grade_avg["Grado"] = grade_avg["Grado_num"].map(
        lambda g: f"{int(g)}°"
    )
    grade_avg["Categoría LATAM"] = (
        grade_avg["Categoria_LATAM"].astype(str)
    )
    grade_avg["Promedio EFA"] = grade_avg["Promedio_EFA"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
    )

    st.markdown("### Promedio EFA por grado y categoría LATAM")
    st.dataframe(
        grade_avg[
            ["Grado", "Categoría LATAM", "n", "Promedio EFA"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    fig_grade = px.line(
        grade_avg,
        x="Categoria_LATAM",
        y="Promedio_EFA",
        color="Grado",
        markers=True,
        text=grade_avg["Promedio_EFA"].map(
            lambda x: f"{x:.0f}%"
        ),
        category_orders={"Categoria_LATAM": LATAM_ORDER},
        custom_data=["n"],
        labels={
            "Categoria_LATAM": "Categoría LATAM",
            "Promedio_EFA": "Promedio EFA",
        },
        title="Promedio EFA por categoría LATAM y grado",
    )
    fig_grade.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
        title="% promedio de aciertos EFA",
    )
    fig_grade.update_traces(
        textposition="top center",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Categoría LATAM: %{x}<br>"
            "Promedio EFA: %{y:.1f}%<br>"
            "Estudiantes: %{customdata[0]}<extra></extra>"
        ),
    )
    fig_grade.update_layout(
        height=540,
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Grado",
    )
    st.plotly_chart(fig_grade, use_container_width=True)

    # La distribución por niveles queda disponible como análisis secundario.
    with st.expander(
        "Ver distribución de niveles EFA dentro de cada categoría LATAM",
        expanded=False,
    ):
        level_counts = (
            filtered.groupby(
                ["Categoria_LATAM", "Nivel_EFA"],
                as_index=False,
            )["Estudiantes"]
            .sum()
        )
        level_counts["Total_categoria"] = (
            level_counts.groupby("Categoria_LATAM")["Estudiantes"]
            .transform("sum")
        )
        level_counts["Porcentaje"] = np.where(
            level_counts["Total_categoria"] > 0,
            level_counts["Estudiantes"]
            / level_counts["Total_categoria"] * 100,
            0,
        )
        level_counts["Categoria_LATAM"] = pd.Categorical(
            level_counts["Categoria_LATAM"],
            categories=LATAM_ORDER,
            ordered=True,
        )
        level_counts = level_counts.sort_values(
            ["Categoria_LATAM", "Nivel_EFA"]
        )

        fig_levels = px.bar(
            level_counts,
            x="Categoria_LATAM",
            y="Porcentaje",
            color="Nivel_EFA",
            barmode="stack",
            category_orders={
                "Categoria_LATAM": LATAM_ORDER,
                "Nivel_EFA": LEVEL_ORDER,
            },
            color_discrete_map=LEVEL_COLORS,
            text=level_counts["Porcentaje"].map(
                lambda x: f"{x:.0f}%" if x >= 5 else ""
            ),
            custom_data=["Estudiantes"],
            labels={
                "Categoria_LATAM": "Categoría LATAM",
                "Porcentaje": "% de estudiantes",
                "Nivel_EFA": "Nivel EFA",
            },
            title=(
                "Distribución de niveles EFA dentro de las categorías LATAM"
            ),
        )
        fig_levels.update_yaxes(range=[0, 100], ticksuffix="%")
        fig_levels.update_traces(
            textposition="inside",
            hovertemplate=(
                "Categoría LATAM: %{x}<br>"
                "Nivel EFA: %{fullData.name}<br>"
                "Estudiantes: %{customdata[0]}<br>"
                "Porcentaje: %{y:.1f}%<extra></extra>"
            ),
        )
        fig_levels.update_layout(
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Nivel EFA",
        )
        st.plotly_chart(fig_levels, use_container_width=True)


def _math_activity_filter(
    df: pd.DataFrame,
    selected_sites: list[str],
    selected_grades: list[int],
) -> pd.DataFrame:
    out = df.copy()
    if selected_sites:
        out = out[out["Sede"].astype(str).isin(selected_sites)]
    if selected_grades:
        out = out[out["Grado_num"].isin(selected_grades)]
    return out


def _math_metric_summary(
    stats: pd.DataFrame,
    metric: str,
    selected_sites: list[str],
    selected_grades: list[int],
) -> dict:
    d = stats[stats["Metrica"].astype(str) == metric].copy()
    d = _math_activity_filter(d, selected_sites, selected_grades)
    if d.empty:
        return {
            "n_total": 0,
            "n": 0,
            "n_na": 0,
            "mean_x": np.nan,
            "mean_efa": np.nan,
            "r": np.nan,
        }

    n = float(d["N_validos"].sum())
    n_total = float(d["N_total"].sum())
    sx = float(d["Sum_X"].sum())
    sy = float(d["Sum_Y"].sum())
    sx2 = float(d["Sum_X2"].sum())
    sy2 = float(d["Sum_Y2"].sum())
    sxy = float(d["Sum_XY"].sum())

    mean_x = sx / n if n > 0 else np.nan
    mean_y = sy / n if n > 0 else np.nan

    if n > 1:
        num = n * sxy - sx * sy
        den_x = n * sx2 - sx * sx
        den_y = n * sy2 - sy * sy
        den = np.sqrt(max(0, den_x) * max(0, den_y))
        r = num / den if den > 0 else np.nan
    else:
        r = np.nan

    return {
        "n_total": int(n_total),
        "n": int(n),
        "n_na": int(max(0, n_total - n)),
        "mean_x": mean_x,
        "mean_efa": mean_y,
        "r": r,
    }


def _math_metric_bins(
    bins: pd.DataFrame,
    metric: str,
    selected_sites: list[str],
    selected_grades: list[int],
) -> pd.DataFrame:
    d = bins[bins["Metrica"].astype(str) == metric].copy()
    d = _math_activity_filter(d, selected_sites, selected_grades)
    if d.empty:
        return pd.DataFrame()

    out = (
        d.groupby(["Rango", "Orden"], as_index=False)
        .agg(
            n=("N_validos", "sum"),
            Sum_X=("Sum_X", "sum"),
            Sum_EFA=("Sum_EFA", "sum"),
        )
        .sort_values("Orden")
    )
    out["Promedio_X"] = np.where(
        out["n"] > 0,
        out["Sum_X"] / out["n"],
        np.nan,
    )
    out["Promedio_EFA"] = np.where(
        out["n"] > 0,
        out["Sum_EFA"] / out["n"],
        np.nan,
    )
    return out


def _math_metric_by_grade(
    stats: pd.DataFrame,
    metric: str,
    selected_sites: list[str],
    selected_grades: list[int],
) -> pd.DataFrame:
    rows = []
    for grade in sorted(selected_grades):
        info = _math_metric_summary(
            stats,
            metric,
            selected_sites,
            [grade],
        )
        if info["n_total"] == 0:
            continue
        rows.append({
            "Grado_num": int(grade),
            "Grado": f"{int(grade)}°",
            "n total": info["n_total"],
            "n con dato": info["n"],
            "NA": info["n_na"],
            "Promedio_variable": info["mean_x"],
            "Promedio_EFA": info["mean_efa"],
            "r_Pearson": info["r"],
        })
    return pd.DataFrame(rows)


def _metric_value_label(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric == "Actividades realizadas":
        return f"{value:.1f}"
    if metric == "Tiempo total (horas)":
        return f"{value:.2f} h"
    return f"{value:.2f}/10"


def _render_math_metric_relation(
    stats: pd.DataFrame,
    bins: pd.DataFrame,
    metric: str,
    selected_sites: list[str],
    selected_grades: list[int],
    title: str,
):
    info = _math_metric_summary(
        stats,
        metric,
        selected_sites,
        selected_grades,
    )

    st.markdown(f"### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Promedio · {metric}",
        _metric_value_label(metric, info["mean_x"]),
    )
    c2.metric(
        "Promedio EFA Matemáticas",
        (
            f"{info['mean_efa']:.1f}%"
            if np.isfinite(info["mean_efa"])
            else "—"
        ),
    )
    c3.metric(
        "Estudiantes con dato",
        f"{info['n']:,}".replace(",", "."),
        help=(
            f"NA excluidos en esta variable: "
            f"{info['n_na']:,}".replace(",", ".")
        ),
    )
    c4.metric(
        "r Pearson",
        f"{info['r']:.2f}" if np.isfinite(info["r"]) else "—",
        help=(
            "Asociación lineal con el porcentaje de aciertos de EFA. "
            "Los NA no participan en el cálculo."
        ),
    )

    plot = _math_metric_bins(
        bins,
        metric,
        selected_sites,
        selected_grades,
    )
    if plot.empty:
        st.info("No hay datos suficientes para mostrar los rangos.")
        return

    fig = px.bar(
        plot,
        x="Rango",
        y="Promedio_EFA",
        text=plot["Promedio_EFA"].map(
            lambda x: f"{x:.1f}%" if pd.notna(x) else ""
        ),
        custom_data=["n", "Promedio_X"],
        labels={
            "Rango": metric,
            "Promedio_EFA": "Promedio EFA",
        },
        title=f"Promedio EFA según {metric.lower()}",
    )
    fig.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
        title="% promedio de aciertos EFA",
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Promedio EFA: %{y:.1f}%<br>"
            "Estudiantes: %{customdata[0]:.0f}<br>"
            "Promedio de la variable: %{customdata[1]:.2f}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        height=480,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"math_metric_bins_{metric}",
    )

    st.markdown("#### Comportamiento por grado")
    by_grade = _math_metric_by_grade(
        stats,
        metric,
        selected_sites,
        selected_grades,
    )

    if by_grade.empty:
        st.info("No hay datos por grado para esta selección.")
        return

    grade_display = by_grade.copy()
    grade_display["Promedio de la variable"] = grade_display[
        "Promedio_variable"
    ].map(lambda x: _metric_value_label(metric, x))
    grade_display["Promedio EFA"] = grade_display[
        "Promedio_EFA"
    ].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
    )
    grade_display["r Pearson"] = grade_display[
        "r_Pearson"
    ].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )

    st.dataframe(
        grade_display[
            [
                "Grado",
                "n total",
                "n con dato",
                "NA",
                "Promedio de la variable",
                "Promedio EFA",
                "r Pearson",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    g1, g2 = st.columns(2)

    with g1:
        fig_var = px.line(
            by_grade,
            x="Grado_num",
            y="Promedio_variable",
            markers=True,
            text=by_grade["Promedio_variable"].map(
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else ""
                )
            ),
            custom_data=["n con dato", "NA"],
            labels={
                "Grado_num": "Grado",
                "Promedio_variable": metric,
            },
            title=f"{metric} por grado",
        )
        fig_var = style_clean_axes(
            fig_var,
            grades=by_grade["Grado_num"].tolist(),
            hide_y=False,
        )
        if metric not in [
            "Actividades realizadas",
            "Tiempo total (horas)",
        ]:
            fig_var.update_yaxes(
                range=[0, 10],
                title="Promedio (0–10)",
            )
        else:
            fig_var.update_yaxes(title=metric)
        fig_var.update_traces(
            textposition="top center",
            cliponaxis=False,
            hovertemplate=(
                "Grado: %{x}°<br>"
                "Promedio: %{y:.2f}<br>"
                "n con dato: %{customdata[0]:.0f}<br>"
                "NA: %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        )
        fig_var.update_layout(height=420)
        st.plotly_chart(
            fig_var,
            use_container_width=True,
            key=f"math_metric_grade_variable_{metric}",
        )

    with g2:
        fig_efa = px.line(
            by_grade,
            x="Grado_num",
            y="Promedio_EFA",
            markers=True,
            text=by_grade["Promedio_EFA"].map(
                lambda x: f"{x:.1f}%" if pd.notna(x) else ""
            ),
            custom_data=["n con dato"],
            labels={
                "Grado_num": "Grado",
                "Promedio_EFA": "Promedio EFA",
            },
            title="Promedio EFA de los mismos estudiantes por grado",
        )
        fig_efa = style_clean_axes(
            fig_efa,
            grades=by_grade["Grado_num"].tolist(),
            hide_y=False,
        )
        fig_efa.update_yaxes(
            range=[0, 100],
            ticksuffix="%",
            title="% promedio de aciertos EFA",
        )
        fig_efa.update_traces(
            textposition="top center",
            cliponaxis=False,
            hovertemplate=(
                "Grado: %{x}°<br>"
                "Promedio EFA: %{y:.1f}%<br>"
                "n con dato: %{customdata[0]:.0f}"
                "<extra></extra>"
            ),
        )
        fig_efa.update_layout(height=420)
        st.plotly_chart(
            fig_efa,
            use_container_width=True,
            key=f"math_metric_grade_efa_{metric}",
        )


def render_math_activity_vs_efa(
    stats: pd.DataFrame,
    bins: pd.DataFrame,
    match_summary: pd.DataFrame,
):
    st.markdown("## Actividad matemática vs Evaluar para Avanzar")
    st.caption(
        "Contraste entre uso/desempeño en actividades matemáticas y "
        "el porcentaje de aciertos en Evaluar para Avanzar."
    )

    if stats.empty or bins.empty:
        st.warning(
            "No está disponible el resumen para esta comparación."
        )
        return

    st.info(
        "Los promedios de la nueva tabla están en escala 0–10. "
        "Una celda vacía se trata como NA y se excluye del promedio y de "
        "la asociación; un 0 registrado sí se conserva como un puntaje válido."
    )

    available_sites = [
        site for site in SITES
        if site in stats["Sede"].dropna().astype(str).unique()
    ]
    grades = sorted(
        [int(g) for g in stats["Grado_num"].dropna().unique()]
    )

    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        selected_sites = st.multiselect(
            "Sedes/colegios",
            available_sites,
            default=available_sites,
            key="math_activity_sites",
        )
    with c2:
        selected_grades = st.multiselect(
            "Grados",
            grades,
            default=grades,
            format_func=lambda g: f"{g}°",
            key="math_activity_grades",
        )

    if not selected_sites or not selected_grades:
        st.warning("Selecciona al menos una sede y un grado.")
        return

    base = _math_metric_summary(
        stats,
        "Actividades realizadas",
        selected_sites,
        selected_grades,
    )
    time_info = _math_metric_summary(
        stats,
        "Tiempo total (horas)",
        selected_sites,
        selected_grades,
    )
    activity_info = base

    coverage = np.nan
    if match_summary is not None and not match_summary.empty:
        coverage = pd.to_numeric(
            match_summary.iloc[0].get(
                "Cobertura_sobre_elegibles_pct",
                np.nan,
            ),
            errors="coerce",
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Estudiantes emparejados",
        f"{base['n_total']:,}".replace(",", "."),
    )
    k2.metric(
        "Promedio EFA Matemáticas",
        (
            f"{base['mean_efa']:.1f}%"
            if np.isfinite(base["mean_efa"])
            else "—"
        ),
    )
    k3.metric(
        "Actividades promedio",
        (
            f"{activity_info['mean_x']:.1f}"
            if np.isfinite(activity_info["mean_x"])
            else "—"
        ),
    )
    k4.metric(
        "Tiempo promedio",
        (
            f"{time_info['mean_x']:.2f} h"
            if np.isfinite(time_info["mean_x"])
            else "—"
        ),
        help=(
            f"Cobertura global del cruce sobre estudiantes elegibles: "
            f"{coverage:.1f}%"
            if np.isfinite(coverage)
            else None
        ),
    )

    st.divider()
    _render_math_metric_relation(
        stats,
        bins,
        "Tiempo total (horas)",
        selected_sites,
        selected_grades,
        "Tiempo destinado a actividades",
    )

    st.divider()
    _render_math_metric_relation(
        stats,
        bins,
        "Actividades realizadas",
        selected_sites,
        selected_grades,
        "Cantidad de actividades realizadas",
    )

    st.divider()
    _render_math_metric_relation(
        stats,
        bins,
        "Promedio general",
        selected_sites,
        selected_grades,
        "Promedio general (0–10)",
    )

    st.divider()
    st.markdown("### Promedio por competencias")
    comp_rows = []
    for metric in MATH_COMPETENCIES:
        info = _math_metric_summary(
            stats,
            metric,
            selected_sites,
            selected_grades,
        )
        coverage_metric = (
            info["n"] / info["n_total"] * 100
            if info["n_total"] > 0
            else np.nan
        )
        comp_rows.append({
            "Competencia": metric,
            "n con dato": info["n"],
            "% con dato": coverage_metric,
            "Promedio (0–10)": info["mean_x"],
            "Promedio EFA": info["mean_efa"],
            "r Pearson": info["r"],
        })

    comp_table = pd.DataFrame(comp_rows)
    comp_display = comp_table.copy()
    comp_display["% con dato"] = comp_display["% con dato"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
    )
    comp_display["Promedio (0–10)"] = comp_display[
        "Promedio (0–10)"
    ].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    comp_display["Promedio EFA"] = comp_display[
        "Promedio EFA"
    ].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
    )
    comp_display["r Pearson"] = comp_display["r Pearson"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )

    st.dataframe(
        comp_display,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "El porcentaje con dato permite ver la cobertura real de cada "
        "competencia. Los estudiantes con celda vacía no se convierten a 0."
    )

    st.markdown("#### Competencias por grado")
    comp_grade_rows = []
    for grade in sorted(selected_grades):
        for metric in MATH_COMPETENCIES:
            info = _math_metric_summary(
                stats,
                metric,
                selected_sites,
                [grade],
            )
            if info["n_total"] == 0:
                continue
            comp_grade_rows.append({
                "Grado_num": int(grade),
                "Grado": f"{int(grade)}°",
                "Competencia": metric,
                "n total": info["n_total"],
                "n con dato": info["n"],
                "NA": info["n_na"],
                "% con dato": (
                    info["n"] / info["n_total"] * 100
                    if info["n_total"] > 0
                    else np.nan
                ),
                "Promedio_competencia": info["mean_x"],
                "Promedio_EFA": info["mean_efa"],
                "r_Pearson": info["r"],
            })

    comp_grade = pd.DataFrame(comp_grade_rows)

    if not comp_grade.empty:
        comp_grade_display = comp_grade.copy()
        comp_grade_display["% con dato"] = comp_grade_display[
            "% con dato"
        ].map(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
        )
        comp_grade_display["Promedio competencia (0–10)"] = (
            comp_grade_display["Promedio_competencia"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—"
            )
        )
        comp_grade_display["Promedio EFA"] = comp_grade_display[
            "Promedio_EFA"
        ].map(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
        )
        comp_grade_display["r Pearson"] = comp_grade_display[
            "r_Pearson"
        ].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—"
        )

        st.dataframe(
            comp_grade_display[
                [
                    "Grado",
                    "Competencia",
                    "n con dato",
                    "NA",
                    "% con dato",
                    "Promedio competencia (0–10)",
                    "Promedio EFA",
                    "r Pearson",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        comp_matrix = (
            comp_grade.pivot(
                index="Competencia",
                columns="Grado_num",
                values="Promedio_competencia",
            )
            .reindex(index=MATH_COMPETENCIES)
        )
        efa_matrix = (
            comp_grade.pivot(
                index="Competencia",
                columns="Grado_num",
                values="Promedio_EFA",
            )
            .reindex(index=MATH_COMPETENCIES)
        )

        h1, h2 = st.columns(2)

        with h1:
            fig_comp = go.Figure(
                data=go.Heatmap(
                    z=comp_matrix.values,
                    x=[f"{int(g)}°" for g in comp_matrix.columns],
                    y=comp_matrix.index,
                    zmin=0,
                    zmax=10,
                    text=np.where(
                        pd.isna(comp_matrix.values),
                        "",
                        np.vectorize(lambda x: f"{x:.1f}")(
                            np.nan_to_num(comp_matrix.values)
                        ),
                    ),
                    texttemplate="%{text}",
                    hovertemplate=(
                        "Competencia: %{y}<br>"
                        "Grado: %{x}<br>"
                        "Promedio: %{z:.2f}/10"
                        "<extra></extra>"
                    ),
                    colorbar=dict(title="0–10"),
                )
            )
            fig_comp.update_layout(
                title="Promedio de competencia por grado",
                height=430,
                margin=dict(l=15, r=15, t=60, b=20),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with h2:
            fig_comp_efa = go.Figure(
                data=go.Heatmap(
                    z=efa_matrix.values,
                    x=[f"{int(g)}°" for g in efa_matrix.columns],
                    y=efa_matrix.index,
                    zmin=0,
                    zmax=100,
                    text=np.where(
                        pd.isna(efa_matrix.values),
                        "",
                        np.vectorize(lambda x: f"{x:.0f}%")(
                            np.nan_to_num(efa_matrix.values)
                        ),
                    ),
                    texttemplate="%{text}",
                    hovertemplate=(
                        "Competencia: %{y}<br>"
                        "Grado: %{x}<br>"
                        "Promedio EFA: %{z:.1f}%"
                        "<extra></extra>"
                    ),
                    colorbar=dict(title="% EFA"),
                )
            )
            fig_comp_efa.update_layout(
                title="Promedio EFA de estudiantes con dato por grado",
                height=430,
                margin=dict(l=15, r=15, t=60, b=20),
            )
            st.plotly_chart(fig_comp_efa, use_container_width=True)

    selected_comp = st.selectbox(
        "Explorar una competencia",
        MATH_COMPETENCIES,
        key="math_activity_competency",
    )
    _render_math_metric_relation(
        stats,
        bins,
        selected_comp,
        selected_sites,
        selected_grades,
        f"Detalle · {selected_comp}",
    )


def _evolution_filter(
    data: pd.DataFrame,
    metric: str | None = None,
    selected_grades: list[int] | None = None,
    selected_sites: list[str] | None = None,
) -> pd.DataFrame:
    d = data.copy()
    if metric is not None and "Metrica" in d.columns:
        d = d[d["Metrica"].astype(str) == metric]
    if selected_grades:
        d = d[d["Grado_num"].isin(selected_grades)]
    if selected_sites and "Sede" in d.columns:
        d = d[d["Sede"].astype(str).isin(selected_sites)]
    return d.copy()


def _evolution_combine(data: pd.DataFrame) -> dict:
    if data is None or data.empty:
        return {
            "n": 0,
            "initial": np.nan,
            "final": np.nan,
            "change": np.nan,
            "improved": np.nan,
            "same": np.nan,
            "decreased": np.nan,
        }

    n = pd.to_numeric(data["N"], errors="coerce").fillna(0).sum()
    if n <= 0:
        return {
            "n": 0,
            "initial": np.nan,
            "final": np.nan,
            "change": np.nan,
            "improved": np.nan,
            "same": np.nan,
            "decreased": np.nan,
        }

    sum_initial = pd.to_numeric(
        data["Sum_Inicial"], errors="coerce"
    ).fillna(0).sum()
    sum_final = pd.to_numeric(
        data["Sum_Final"], errors="coerce"
    ).fillna(0).sum()
    sum_change = pd.to_numeric(
        data["Sum_Cambio"], errors="coerce"
    ).fillna(0).sum()
    n_improved = pd.to_numeric(
        data["N_Mejora"], errors="coerce"
    ).fillna(0).sum()
    n_same = pd.to_numeric(
        data["N_Igual"], errors="coerce"
    ).fillna(0).sum()
    n_down = pd.to_numeric(
        data["N_Baja"], errors="coerce"
    ).fillna(0).sum()

    return {
        "n": int(n),
        "initial": sum_initial / n,
        "final": sum_final / n,
        "change": sum_change / n,
        "improved": n_improved / n * 100,
        "same": n_same / n * 100,
        "decreased": n_down / n * 100,
    }


def _evolution_grouped(
    data: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    if data.empty:
        return pd.DataFrame()

    for keys, group in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        info = _evolution_combine(group)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "N": info["n"],
            "Promedio inicial": info["initial"],
            "Promedio final": info["final"],
            "Cambio promedio": info["change"],
            "% mejora": info["improved"],
            "% igual": info["same"],
            "% baja": info["decreased"],
        })
        rows.append(row)

    return pd.DataFrame(rows)


def _observed_value(
    observed: pd.DataFrame,
    grade: int,
    site: str,
    moment: str,
    metric: str,
) -> dict:
    d = observed[
        (observed["Grado_num"] == grade)
        & (observed["Sede"].astype(str) == site)
        & (observed["Momento"].astype(str) == moment)
        & (observed["Metrica"].astype(str) == metric)
    ].copy()
    if d.empty:
        return {"n": 0, "mean": np.nan}
    row = d.iloc[0]
    return {
        "n": int(pd.to_numeric(row["N"], errors="coerce")),
        "mean": float(pd.to_numeric(row["Promedio"], errors="coerce")),
    }


def _paired_site_table(
    summary: pd.DataFrame,
    grade: int,
    metric: str,
) -> pd.DataFrame:
    d = summary[
        (summary["Grado_num"] == grade)
        & (summary["Metrica"].astype(str) == metric)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for _, r in d.iterrows():
        n = float(r["N"])
        if n <= 0:
            continue
        initial = float(r["Sum_Inicial"]) / n
        final = float(r["Sum_Final"]) / n
        change = float(r["Sum_Cambio"]) / n
        improved = float(r["N_Mejora"]) / n * 100
        rows.append({
            "Sede": r["Sede"],
            "n pareados": int(n),
            "Primera aplicación": initial,
            "Última aplicación": final,
            "Cambio pareado": change,
            "% mejora": improved,
        })
    return pd.DataFrame(rows)


def render_evolution_10_11(
    summary: pd.DataFrame,
    trajectory: pd.DataFrame,
    availability: pd.DataFrame,
    observed: pd.DataFrame,
):
    st.markdown("## Evolución de resultados · 10° y 11°")
    st.caption(
        "Evolución de puntajes entre aplicaciones, con comparación "
        "pareada de los mismos estudiantes."
    )

    if summary.empty or observed.empty:
        st.warning("No está disponible la información de evolución.")
        return

    st.info(
        "El crecimiento se calcula comparando a los mismos estudiantes "
        "entre aplicaciones. Cota se muestra cuando tiene información, "
        "aunque no tenga una segunda aplicación para calcular cambio."
    )

    tab10, tab11 = st.tabs(
        ["10° · Crecimiento", "11° · Mosquera"]
    )

    with tab10:
        st.markdown("### Crecimiento global · mismos estudiantes")
        st.caption(
            "Compara Evaluación 1 vs Evaluación 2 únicamente en estudiantes "
            "que presentaron ambas aplicaciones."
        )

        paired10 = _paired_site_table(
            summary,
            10,
            "Puntaje global",
        )

        if not paired10.empty:
            overall10 = _evolution_combine(
                summary[
                    (summary["Grado_num"] == 10)
                    & (
                        summary["Metrica"].astype(str)
                        == "Puntaje global"
                    )
                ]
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Estudiantes pareados",
                f"{overall10['n']:,}".replace(",", "."),
            )
            k2.metric(
                "Primera aplicación · pareados",
                f"{overall10['initial']:.1f}",
            )
            k3.metric(
                "Segunda aplicación · pareados",
                f"{overall10['final']:.1f}",
            )
            k4.metric(
                "Crecimiento promedio",
                f"{overall10['change']:+.1f}",
            )

            long10 = pd.concat(
                [
                    paired10[
                        ["Sede", "Primera aplicación"]
                    ]
                    .rename(
                        columns={"Primera aplicación": "Promedio"}
                    )
                    .assign(Evaluación="Evaluación 1"),
                    paired10[
                        ["Sede", "Última aplicación"]
                    ]
                    .rename(
                        columns={"Última aplicación": "Promedio"}
                    )
                    .assign(Evaluación="Evaluación 2"),
                ],
                ignore_index=True,
            )

            # Cota solo cuenta con Evaluación 1. Se muestra para dar
            # visibilidad a la sede, pero no entra al cálculo de crecimiento.
            cota_ev1 = _observed_value(
                observed,
                10,
                "Cota",
                "Evaluación 1",
                "Puntaje global",
            )
            if cota_ev1["n"] > 0 and np.isfinite(cota_ev1["mean"]):
                long10 = pd.concat(
                    [
                        long10,
                        pd.DataFrame([{
                            "Sede": "Cota",
                            "Promedio": cota_ev1["mean"],
                            "Evaluación": "Evaluación 1",
                        }]),
                    ],
                    ignore_index=True,
                )

            fig10_growth = px.bar(
                long10,
                x="Sede",
                y="Promedio",
                color="Evaluación",
                barmode="group",
                text=long10["Promedio"].map(
                    lambda x: f"{x:.1f}"
                ),
                title=(
                    "Puntaje global · Evaluación 1 vs Evaluación 2 "
                    "(Cota solo cuenta con Evaluación 1)"
                ),
                category_orders={
                    "Sede": ["Cota", "Mosquera", "Tunja", "Zipaquirá"]
                },
            )
            fig10_growth.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig10_growth.update_layout(
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Aplicación",
            )
            st.plotly_chart(
                fig10_growth,
                use_container_width=True,
                key="evolution10_paired_global",
            )

            growth_display = paired10.copy()
            for col in [
                "Primera aplicación",
                "Última aplicación",
            ]:
                growth_display[col] = growth_display[col].map(
                    lambda x: f"{x:.1f}"
                )
            growth_display["Cambio pareado"] = growth_display[
                "Cambio pareado"
            ].map(lambda x: f"{x:+.1f}")
            growth_display["% mejora"] = growth_display[
                "% mejora"
            ].map(lambda x: f"{x:.1f}%")

            cota_status = pd.DataFrame([{
                "Sede": "Cota",
                "n pareados": "—",
                "Primera aplicación": (
                    f"{cota_ev1['mean']:.1f}"
                    if cota_ev1["n"] > 0
                    and np.isfinite(cota_ev1["mean"])
                    else "Sin dato"
                ),
                "Última aplicación": "Sin aplicación 2",
                "Cambio pareado": "—",
                "% mejora": "—",
            }])
            growth_display = pd.concat(
                [growth_display, cota_status],
                ignore_index=True,
            )

            st.dataframe(
                growth_display[
                    [
                        "Sede",
                        "n pareados",
                        "Primera aplicación",
                        "Última aplicación",
                        "Cambio pareado",
                        "% mejora",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            best10 = paired10.sort_values(
                "Cambio pareado",
                ascending=False,
            ).iloc[0]
            st.success(
                f"Mayor crecimiento en 10°: **{best10['Sede']}**, "
                f"con {best10['Cambio pareado']:+.1f} puntos en promedio."
            )
            st.caption(
                "Cota aparece explícitamente, pero no tiene Evaluación 2; "
                "por eso no puede calcularse crecimiento pareado."
            )

        st.divider()
        st.markdown("### Comportamiento por cada prueba")
        st.caption(
            "Cada prueba compara Evaluación 1 vs Evaluación 2 por sede. "
            "El cambio se calcula únicamente con estudiantes que presentaron "
            "ambas aplicaciones."
        )

        subject_metrics_10 = [
            ("Matemáticas", "Puntaje Matemáticas"),
            ("Lenguaje", "Puntaje Lectura crítica"),
            ("Ciencias naturales", "Puntaje Ciencias naturales"),
            ("Ciencias sociales", "Puntaje Sociales y ciudadanas"),
            ("Inglés", "Puntaje Inglés"),
        ]

        subject_tabs_10 = st.tabs(
            [label for label, _ in subject_metrics_10]
        )

        for subject_tab, (subject_label, subject_metric) in zip(
            subject_tabs_10,
            subject_metrics_10,
        ):
            with subject_tab:
                paired_subject = _paired_site_table(
                    summary,
                    10,
                    subject_metric,
                )

                subject_overall = _evolution_combine(
                    summary[
                        (summary["Grado_num"] == 10)
                        & (
                            summary["Metrica"].astype(str)
                            == subject_metric
                        )
                    ]
                )

                s1, s2, s3, s4 = st.columns(4)
                s1.metric(
                    "Estudiantes pareados",
                    f"{subject_overall['n']:,}".replace(",", "."),
                )
                s2.metric(
                    "Evaluación 1 · pareados",
                    f"{subject_overall['initial']:.1f}",
                )
                s3.metric(
                    "Evaluación 2 · pareados",
                    f"{subject_overall['final']:.1f}",
                )
                s4.metric(
                    "Cambio promedio",
                    f"{subject_overall['change']:+.1f}",
                )

                if paired_subject.empty:
                    st.info(
                        "No hay estudiantes pareados para esta prueba."
                    )
                    continue

                subject_long = pd.concat(
                    [
                        paired_subject[
                            ["Sede", "Primera aplicación"]
                        ]
                        .rename(
                            columns={
                                "Primera aplicación": "Promedio"
                            }
                        )
                        .assign(Evaluación="Evaluación 1"),
                        paired_subject[
                            ["Sede", "Última aplicación"]
                        ]
                        .rename(
                            columns={
                                "Última aplicación": "Promedio"
                            }
                        )
                        .assign(Evaluación="Evaluación 2"),
                    ],
                    ignore_index=True,
                )

                cota_subject_ev1 = _observed_value(
                    observed,
                    10,
                    "Cota",
                    "Evaluación 1",
                    subject_metric,
                )
                if (
                    cota_subject_ev1["n"] > 0
                    and np.isfinite(cota_subject_ev1["mean"])
                ):
                    subject_long = pd.concat(
                        [
                            subject_long,
                            pd.DataFrame([{
                                "Sede": "Cota",
                                "Promedio": cota_subject_ev1["mean"],
                                "Evaluación": "Evaluación 1",
                            }]),
                        ],
                        ignore_index=True,
                    )

                fig_subject_10 = px.bar(
                    subject_long,
                    x="Sede",
                    y="Promedio",
                    color="Evaluación",
                    barmode="group",
                    text=subject_long["Promedio"].map(
                        lambda x: f"{x:.1f}"
                    ),
                    category_orders={
                        "Sede": [
                            "Cota",
                            "Mosquera",
                            "Tunja",
                            "Zipaquirá",
                        ]
                    },
                    title=(
                        f"{subject_label} · Evaluación 1 vs "
                        "Evaluación 2"
                    ),
                )
                fig_subject_10.update_yaxes(
                    range=[0, 100],
                    title="Puntaje promedio",
                )
                fig_subject_10.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )
                fig_subject_10.update_layout(
                    height=490,
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Aplicación",
                )
                st.plotly_chart(
                    fig_subject_10,
                    use_container_width=True,
                    key=(
                        "evolution10_subject_"
                        + subject_metric.replace(" ", "_")
                    ),
                )

                subject_display = paired_subject.copy()
                for col in [
                    "Primera aplicación",
                    "Última aplicación",
                ]:
                    subject_display[col] = subject_display[col].map(
                        lambda x: f"{x:.1f}"
                    )
                subject_display["Cambio pareado"] = subject_display[
                    "Cambio pareado"
                ].map(lambda x: f"{x:+.1f}")
                subject_display["% mejora"] = subject_display[
                    "% mejora"
                ].map(lambda x: f"{x:.1f}%")

                cota_subject_status = pd.DataFrame([{
                    "Sede": "Cota",
                    "n pareados": "—",
                    "Primera aplicación": (
                        f"{cota_subject_ev1['mean']:.1f}"
                        if cota_subject_ev1["n"] > 0
                        and np.isfinite(cota_subject_ev1["mean"])
                        else "Sin dato"
                    ),
                    "Última aplicación": "Sin aplicación 2",
                    "Cambio pareado": "—",
                    "% mejora": "—",
                }])

                subject_display = pd.concat(
                    [subject_display, cota_subject_status],
                    ignore_index=True,
                )

                st.dataframe(
                    subject_display[
                        [
                            "Sede",
                            "n pareados",
                            "Primera aplicación",
                            "Última aplicación",
                            "Cambio pareado",
                            "% mejora",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                ranked_subject = paired_subject.sort_values(
                    "Cambio pareado",
                    ascending=False,
                )
                if not ranked_subject.empty:
                    best_subject = ranked_subject.iloc[0]
                    st.caption(
                        f"Mayor avance en {subject_label}: "
                        f"**{best_subject['Sede']}** "
                        f"({best_subject['Cambio pareado']:+.1f} puntos)."
                    )

    with tab11:
        st.markdown("### Mosquera · crecimiento global")
        st.caption(
            "11° tiene tres aplicaciones. La trayectoria se calcula sobre los "
            "51 estudiantes que cuentan con Evaluación 1, 2 y 3."
        )

        pair11 = summary[
            (summary["Grado_num"] == 11)
            & (summary["Sede"].astype(str) == "Mosquera")
        ].copy()
        global11 = _evolution_combine(
            pair11[
                pair11["Metrica"].astype(str) == "Puntaje global"
            ]
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Estudiantes completos", f"{global11['n']}")
        c2.metric("Evaluación 1", f"{global11['initial']:.1f}")
        c3.metric("Evaluación 3", f"{global11['final']:.1f}")
        c4.metric("Crecimiento 1 → 3", f"{global11['change']:+.1f}")
        c5.metric("% que mejora", f"{global11['improved']:.1f}%")

        traj11_global = trajectory[
            (trajectory["Grado_num"] == 11)
            & (trajectory["Sede"].astype(str) == "Mosquera")
            & (
                trajectory["Metrica"].astype(str)
                == "Puntaje global"
            )
        ].copy()
        if not traj11_global.empty:
            traj11_global["Promedio"] = (
                traj11_global["Sum_Puntaje"]
                / traj11_global["N"]
            )
            traj11_global["Evaluación"] = (
                traj11_global["Orden"].map({
                    1: "Evaluación 1",
                    2: "Evaluación 2",
                    3: "Evaluación 3",
                })
            )
            fig11_global = px.bar(
                traj11_global.sort_values("Orden"),
                x="Evaluación",
                y="Promedio",
                text=traj11_global["Promedio"].map(
                    lambda x: f"{x:.1f}"
                ),
                title=(
                    "Puntaje global · mismos 51 estudiantes "
                    "en las tres aplicaciones"
                ),
            )
            fig11_global.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig11_global.update_layout(
                height=470,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig11_global,
                use_container_width=True,
                key="evolution11_global_three_apps",
            )

        st.divider()
        st.markdown("### Mosquera · crecimiento por prueba")

        subjects11 = pair11[
            pair11["Metrica"].astype(str) != "Puntaje global"
        ].copy()
        evolution_subject_labels = {
            "Puntaje Matemáticas": "Matemáticas",
            "Puntaje Lectura crítica": "Lenguaje",
            "Puntaje Ciencias naturales": "Ciencias naturales",
            "Puntaje Sociales y ciudadanas": "Ciencias sociales",
            "Puntaje Inglés": "Inglés",
        }
        subject_rows = []
        for _, r in subjects11.iterrows():
            n = float(r["N"])
            subject_rows.append({
                "Prueba": evolution_subject_labels.get(
                    str(r["Metrica"]),
                    str(r["Metrica"]).replace("Puntaje ", ""),
                ),
                "n": int(n),
                "Evaluación 1": float(r["Sum_Inicial"]) / n,
                "Evaluación 3": float(r["Sum_Final"]) / n,
                "Crecimiento 1 → 3": float(r["Sum_Cambio"]) / n,
                "% mejora": float(r["N_Mejora"]) / n * 100,
            })
        subject11 = pd.DataFrame(subject_rows)

        traj11_subjects = trajectory[
            (trajectory["Grado_num"] == 11)
            & (trajectory["Sede"].astype(str) == "Mosquera")
            & (
                trajectory["Metrica"].astype(str)
                != "Puntaje global"
            )
        ].copy()
        if not traj11_subjects.empty:
            traj11_subjects["Promedio"] = (
                traj11_subjects["Sum_Puntaje"]
                / traj11_subjects["N"]
            )
            traj11_subjects["Prueba"] = (
                traj11_subjects["Metrica"]
                .astype(str)
                .map(evolution_subject_labels)
                .fillna(
                    traj11_subjects["Metrica"]
                    .astype(str)
                    .str.replace("Puntaje ", "", regex=False)
                )
            )
            traj11_subjects["Evaluación"] = (
                traj11_subjects["Orden"].map({
                    1: "Evaluación 1",
                    2: "Evaluación 2",
                    3: "Evaluación 3",
                })
            )

            fig11_subjects = px.bar(
                traj11_subjects.sort_values(
                    ["Prueba", "Orden"]
                ),
                x="Prueba",
                y="Promedio",
                color="Evaluación",
                barmode="group",
                text=traj11_subjects["Promedio"].map(
                    lambda x: f"{x:.1f}"
                ),
                title=(
                    "Puntajes por prueba · Evaluación 1, 2 y 3 "
                    "(mismos 51 estudiantes)"
                ),
            )
            fig11_subjects.update_yaxes(
                range=[0, 100],
                title="Puntaje promedio",
            )
            fig11_subjects.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig11_subjects.update_layout(
                height=560,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Aplicación",
            )
            st.plotly_chart(
                fig11_subjects,
                use_container_width=True,
                key="evolution11_subjects_three_apps",
            )

        if not subject11.empty:
            subject_display = subject11.copy()
            for col in ["Evaluación 1", "Evaluación 3"]:
                subject_display[col] = subject_display[col].map(
                    lambda x: f"{x:.1f}"
                )
            subject_display["Crecimiento 1 → 3"] = subject_display[
                "Crecimiento 1 → 3"
            ].map(lambda x: f"{x:+.1f}")
            subject_display["% mejora"] = subject_display[
                "% mejora"
            ].map(lambda x: f"{x:.1f}%")
            st.dataframe(
                subject_display,
                use_container_width=True,
                hide_index=True,
            )

            fig11_delta = px.bar(
                subject11.sort_values(
                    "Crecimiento 1 → 3",
                    ascending=False,
                ),
                x="Prueba",
                y="Crecimiento 1 → 3",
                text=subject11["Crecimiento 1 → 3"].map(
                    lambda x: f"{x:+.1f}"
                ),
                title=(
                    "Crecimiento promedio por prueba · "
                    "Evaluación 1 → Evaluación 3"
                ),
            )
            fig11_delta.add_hline(y=0, line_dash="dot")
            fig11_delta.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig11_delta.update_layout(
                height=470,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig11_delta,
                use_container_width=True,
                key="evolution11_subject_growth",
            )

def _relation_strength(value: float) -> str:
    if not np.isfinite(value):
        return "Sin dato"
    v = abs(float(value))
    if v < 0.20:
        return "Muy débil"
    if v < 0.40:
        return "Débil"
    if v < 0.60:
        return "Moderada"
    if v < 0.80:
        return "Fuerte"
    return "Muy fuerte"


def render_improve_efa_relation(
    levels: pd.DataFrame,
    quality: pd.DataFrame,
    growth: pd.DataFrame,
):
    st.markdown("## Evaluar para Avanzar vs Improve")
    st.caption(
        "Promedio de la primera aplicación de Improve según el nivel de "
        "desempeño alcanzado por los mismos estudiantes en EFA."
    )

    if levels.empty:
        st.warning(
            "No está disponible el resumen por niveles EFA."
        )
        return

    st.info(
        "El emparejamiento es estricto: documento exacto + mismo grado + "
        "misma sede, excluyendo registros marcados para revisión. "
        "Los niveles EFA son: Nivel 1 = 0–25%, Nivel 2 = >25–50%, "
        "Nivel 3 = >50–75% y Nivel 4 = >75–100%. "
        "Improve corresponde exclusivamente a la primera aplicación."
    )

    level_order = ["Nivel 1", "Nivel 2", "Nivel 3", "Nivel 4"]
    test_order = IMPROVE_EFA_TESTS
    site_order = [
        "Cota",
        "Mosquera",
        "Tunja",
        "Zipaquirá",
        "BAQ",
        "Usaquén",
    ]

    for grade in [10, 11]:
        grade_levels = levels[
            (levels["grado_num"] == grade)
            & levels["Prueba"].astype(str).isin(test_order)
        ].copy()

        if grade_levels.empty:
            continue

        st.divider()
        st.markdown(f"## {grade}°")

        q_grade = (
            quality[
                (quality["grado_num"] == grade)
                & (quality["Sede"].astype(str) == "RED")
            ]
            if quality is not None and not quality.empty
            else pd.DataFrame()
        )

        if not q_grade.empty:
            qrow = q_grade.iloc[0]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric(
                "Estudiantes Improve EV1",
                f"{int(qrow['Improve_EV1_estudiantes'])}",
            )
            q2.metric(
                "Emparejados estrictos",
                f"{int(qrow['Emparejados_identidad'])}",
            )
            q3.metric(
                "Cobertura del cruce",
                f"{float(qrow['Cobertura_identidad_pct']):.1f}%",
            )
            q4.metric(
                "No emparejados",
                f"{int(qrow['No_emparejados'])}",
            )

        st.markdown("### Promedio Improve por nivel de EFA · RED")
        st.caption(
            "La lectura central es vertical: dentro de cada prueba, observa "
            "cómo cambia el promedio de Improve al pasar de Nivel 1 a Nivel 4 "
            "en Evaluar para Avanzar."
        )

        red = grade_levels[
            grade_levels["Sede"].astype(str) == "RED"
        ].copy()
        red["Nivel_EFA"] = pd.Categorical(
            red["Nivel_EFA"],
            categories=level_order,
            ordered=True,
        )
        red["Prueba"] = pd.Categorical(
            red["Prueba"],
            categories=test_order,
            ordered=True,
        )
        red = red.sort_values(["Prueba", "Orden"])

        if not red.empty:
            fig_red = px.line(
                red,
                x="Nivel_EFA",
                y="Promedio_Improve",
                facet_col="Prueba",
                facet_col_wrap=3,
                markers=True,
                text=red["Promedio_Improve"].map(
                    lambda x: f"{x:.1f}"
                ),
                custom_data=["n", "Promedio_EFA"],
                category_orders={
                    "Nivel_EFA": level_order,
                    "Prueba": test_order,
                },
                title=(
                    f"Primera aplicación Improve según nivel EFA · RED, "
                    f"{grade}°"
                ),
            )
            fig_red.update_yaxes(
                range=[0, 100],
                title="Promedio Improve",
            )
            fig_red.update_xaxes(
                title=None,
                categoryorder="array",
                categoryarray=level_order,
            )
            fig_red.update_traces(
                textposition="top center",
                cliponaxis=False,
                hovertemplate=(
                    "Nivel EFA: %{x}<br>"
                    "Promedio Improve: %{y:.1f}<br>"
                    "n: %{customdata[0]}<br>"
                    "Promedio EFA del grupo: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
            fig_red.for_each_annotation(
                lambda a: a.update(
                    text=a.text.replace("Prueba=", "")
                )
            )
            fig_red.update_layout(
                height=820,
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(
                fig_red,
                use_container_width=True,
                key=f"improve_efa_levels_red_{grade}",
            )

            matrix = red.pivot_table(
                index="Prueba",
                columns="Nivel_EFA",
                values="Promedio_Improve",
                aggfunc="first",
                observed=False,
            ).reindex(index=test_order, columns=level_order)

            n_matrix = red.pivot_table(
                index="Prueba",
                columns="Nivel_EFA",
                values="n",
                aggfunc="first",
                observed=False,
            ).reindex(index=test_order, columns=level_order)

            display_rows = []
            for test in test_order:
                row = {"Prueba": test}
                for level in level_order:
                    val = matrix.loc[test, level]
                    n_val = n_matrix.loc[test, level]
                    if pd.isna(val):
                        row[level] = "Sin casos"
                    else:
                        row[level] = (
                            f"{val:.1f} (n={int(n_val)})"
                            if not pd.isna(n_val)
                            else f"{val:.1f}"
                        )
                display_rows.append(row)

            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Comportamiento por sede")
        st.caption(
            "Cada línea corresponde a una sede. Esto permite verificar si la "
            "progresión entre niveles EFA e Improve se mantiene también dentro "
            "de los colegios y no solo en el promedio de la RED."
        )

        sites = grade_levels[
            grade_levels["Sede"].astype(str) != "RED"
        ].copy()
        sites = sites[
            sites["Sede"].astype(str).isin(site_order)
        ]
        sites["Nivel_EFA"] = pd.Categorical(
            sites["Nivel_EFA"],
            categories=level_order,
            ordered=True,
        )
        sites["Prueba"] = pd.Categorical(
            sites["Prueba"],
            categories=test_order,
            ordered=True,
        )
        sites["Sede"] = pd.Categorical(
            sites["Sede"],
            categories=site_order,
            ordered=True,
        )
        sites = sites.sort_values(
            ["Prueba", "Sede", "Orden"]
        )

        if not sites.empty:
            fig_sites = px.line(
                sites,
                x="Nivel_EFA",
                y="Promedio_Improve",
                color="Sede",
                facet_col="Prueba",
                facet_col_wrap=3,
                markers=True,
                custom_data=["n", "Promedio_EFA"],
                category_orders={
                    "Nivel_EFA": level_order,
                    "Prueba": test_order,
                    "Sede": site_order,
                },
                title=(
                    f"Promedio Improve por nivel EFA y sede · {grade}°"
                ),
            )
            fig_sites.update_yaxes(
                range=[0, 100],
                title="Promedio Improve",
            )
            fig_sites.update_xaxes(
                title=None,
                categoryorder="array",
                categoryarray=level_order,
            )
            fig_sites.update_traces(
                hovertemplate=(
                    "Sede: %{fullData.name}<br>"
                    "Nivel EFA: %{x}<br>"
                    "Promedio Improve: %{y:.1f}<br>"
                    "n: %{customdata[0]}<br>"
                    "Promedio EFA del grupo: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
            fig_sites.for_each_annotation(
                lambda a: a.update(
                    text=a.text.replace("Prueba=", "")
                )
            )
            fig_sites.update_layout(
                height=840,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Sede",
            )
            st.plotly_chart(
                fig_sites,
                use_container_width=True,
                key=f"improve_efa_levels_sites_{grade}",
            )

        if growth is not None and not growth.empty:
            st.markdown(
                "### Crecimiento en Improve según nivel de Evaluar para Avanzar"
            )
            final_label = (
                "Evaluación 2"
                if grade == 10
                else "Evaluación 3"
            )
            st.caption(
                f"Se comparan los mismos estudiantes entre Evaluación 1 y "
                f"{final_label} de Improve. Los grupos se forman según el "
                "nivel obtenido en EFA para cada prueba."
            )

            growth_grade = growth[
                (growth["grado_num"] == grade)
                & (growth["Sede"].astype(str) == "RED")
                & growth["Prueba"].astype(str).isin(test_order)
            ].copy()

            if not growth_grade.empty:
                growth_grade["Nivel_EFA"] = pd.Categorical(
                    growth_grade["Nivel_EFA"],
                    categories=level_order,
                    ordered=True,
                )
                growth_grade["Prueba"] = pd.Categorical(
                    growth_grade["Prueba"],
                    categories=test_order,
                    ordered=True,
                )
                growth_grade = growth_grade.sort_values(
                    ["Prueba", "Orden"]
                )

                fig_growth = px.bar(
                    growth_grade,
                    x="Prueba",
                    y="Cambio_Improve",
                    color="Nivel_EFA",
                    barmode="group",
                    text=growth_grade["Cambio_Improve"].map(
                        lambda x: f"{x:+.1f}"
                    ),
                    custom_data=["n", "Improve_inicial", "Improve_final"],
                    category_orders={
                        "Prueba": test_order,
                        "Nivel_EFA": level_order,
                    },
                    title=(
                        f"Cambio promedio en Improve según nivel EFA · "
                        f"{grade}°"
                    ),
                )
                fig_growth.add_hline(y=0, line_dash="dot")
                fig_growth.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "Prueba: %{x}<br>"
                        "Nivel EFA: %{fullData.name}<br>"
                        "Cambio: %{y:+.1f}<br>"
                        "n: %{customdata[0]}<br>"
                        "Inicial: %{customdata[1]:.1f}<br>"
                        "Final: %{customdata[2]:.1f}"
                        "<extra></extra>"
                    ),
                )
                fig_growth.update_layout(
                    height=540,
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Nivel EFA",
                )
                st.plotly_chart(
                    fig_growth,
                    use_container_width=True,
                    key=f"improve_growth_by_efa_level_{grade}",
                )

                st.markdown("#### De cuánto a cuánto cambió cada nivel")

                long_growth = pd.concat(
                    [
                        growth_grade[
                            [
                                "Prueba",
                                "Nivel_EFA",
                                "n",
                                "Improve_inicial",
                            ]
                        ]
                        .rename(
                            columns={
                                "Improve_inicial": "Promedio Improve"
                            }
                        )
                        .assign(Momento="Evaluación 1"),
                        growth_grade[
                            [
                                "Prueba",
                                "Nivel_EFA",
                                "n",
                                "Improve_final",
                            ]
                        ]
                        .rename(
                            columns={
                                "Improve_final": "Promedio Improve"
                            }
                        )
                        .assign(Momento=final_label),
                    ],
                    ignore_index=True,
                )

                fig_initial_final = px.line(
                    long_growth,
                    x="Momento",
                    y="Promedio Improve",
                    color="Nivel_EFA",
                    facet_col="Prueba",
                    facet_col_wrap=3,
                    markers=True,
                    text=long_growth["Promedio Improve"].map(
                        lambda x: f"{x:.1f}"
                    ),
                    custom_data=["n"],
                    category_orders={
                        "Prueba": test_order,
                        "Nivel_EFA": level_order,
                        "Momento": ["Evaluación 1", final_label],
                    },
                    title=(
                        f"Puntaje Improve inicial vs final por nivel EFA · "
                        f"{grade}°"
                    ),
                )
                fig_initial_final.update_yaxes(
                    range=[0, 100],
                    title="Promedio Improve",
                )
                fig_initial_final.update_xaxes(title=None)
                fig_initial_final.update_traces(
                    textposition="top center",
                    cliponaxis=False,
                    hovertemplate=(
                        "Nivel EFA: %{fullData.name}<br>"
                        "Momento: %{x}<br>"
                        "Promedio Improve: %{y:.1f}<br>"
                        "n: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                )
                fig_initial_final.for_each_annotation(
                    lambda a: a.update(
                        text=a.text.replace("Prueba=", "")
                    )
                )
                fig_initial_final.update_layout(
                    height=840,
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Nivel EFA",
                )
                st.plotly_chart(
                    fig_initial_final,
                    use_container_width=True,
                    key=f"improve_initial_final_by_efa_level_{grade}",
                )

                best_rows = []
                for test in test_order:
                    dt = growth_grade[
                        growth_grade["Prueba"].astype(str) == test
                    ].copy()
                    if dt.empty:
                        continue
                    best = dt.sort_values(
                        "Cambio_Improve",
                        ascending=False,
                    ).iloc[0]
                    best_rows.append({
                        "Prueba": test,
                        "Nivel con mejor cambio": str(
                            best["Nivel_EFA"]
                        ),
                        "Cambio promedio": best["Cambio_Improve"],
                        "Inicial": best["Improve_inicial"],
                        "Final": best["Improve_final"],
                        "n": int(best["n"]),
                    })

                if best_rows:
                    best_table = pd.DataFrame(best_rows)
                    best_display = best_table.copy()
                    best_display["Cambio promedio"] = best_display[
                        "Cambio promedio"
                    ].map(lambda x: f"{x:+.1f}")
                    best_display["Inicial"] = best_display[
                        "Inicial"
                    ].map(lambda x: f"{x:.1f}")
                    best_display["Final"] = best_display[
                        "Final"
                    ].map(lambda x: f"{x:.1f}")
                    st.markdown(
                        "#### Nivel EFA con mejor cambio por prueba"
                    )
                    st.dataframe(
                        best_display,
                        use_container_width=True,
                        hide_index=True,
                    )

                growth_display = growth_grade[
                    [
                        "Prueba",
                        "Nivel_EFA",
                        "n",
                        "Improve_inicial",
                        "Improve_final",
                        "Cambio_Improve",
                        "Pct_mejora",
                    ]
                ].copy()
                growth_display = growth_display.rename(
                    columns={
                        "Nivel_EFA": "Nivel EFA",
                        "n": "Estudiantes",
                        "Improve_inicial": "Inicial",
                        "Improve_final": "Final",
                        "Cambio_Improve": "Cambio",
                        "Pct_mejora": "% que mejora",
                    }
                )
                for col in ["Inicial", "Final"]:
                    growth_display[col] = growth_display[col].map(
                        lambda x: f"{x:.1f}"
                    )
                growth_display["Cambio"] = growth_display[
                    "Cambio"
                ].map(lambda x: f"{x:+.1f}")
                growth_display["% que mejora"] = growth_display[
                    "% que mejora"
                ].map(lambda x: f"{x:.1f}%")

                st.dataframe(
                    growth_display,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Los cambios con grupos muy pequeños deben interpretarse "
                    "con cautela. Se muestra el tamaño de cada grupo (n) junto "
                    "con el resultado."
                )

        st.markdown("### Tabla completa · prueba × nivel × sede")
        table = grade_levels[
            grade_levels["Prueba"].astype(str).isin(test_order)
        ].copy()
        table["Nivel_EFA"] = pd.Categorical(
            table["Nivel_EFA"],
            categories=level_order,
            ordered=True,
        )
        table["Prueba"] = pd.Categorical(
            table["Prueba"],
            categories=test_order,
            ordered=True,
        )
        table = table.sort_values(
            ["Sede", "Prueba", "Orden"]
        )

        table_display = table[
            [
                "Sede",
                "Prueba",
                "Nivel_EFA",
                "n",
                "Promedio_EFA",
                "Promedio_Improve",
            ]
        ].copy()
        table_display = table_display.rename(
            columns={
                "Nivel_EFA": "Nivel EFA",
                "n": "Estudiantes",
                "Promedio_EFA": "Promedio EFA",
                "Promedio_Improve": "Promedio Improve",
            }
        )
        table_display["Promedio EFA"] = table_display[
            "Promedio EFA"
        ].map(lambda x: f"{x:.1f}%")
        table_display["Promedio Improve"] = table_display[
            "Promedio Improve"
        ].map(lambda x: f"{x:.1f}/100")

        st.dataframe(
            table_display,
            use_container_width=True,
            hide_index=True,
        )


def render_internal_vs_efa(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    coverage: pd.DataFrame,
    agreement: pd.DataFrame,
    reverse_exact: pd.DataFrame,
):
    st.markdown("## Evaluación interna vs Evaluar para Avanzar")
    st.caption(
        "Comparación entre la evaluación interna y EFA usando únicamente "
        "Trimestre 1 y Trimestre 2."
    )

    if summary.empty:
        st.warning(
            "No está disponible el cruce entre evaluación interna y EFA."
        )
        return

    st.info(
        "La unión se realiza por correo exacto normalizado y se valida grado "
        "y sede. La evaluación interna usa la Escala DESPUÉS DE POTENCIACIÓN. "
        "Esta vista temporal se limita a Trimestre 1 y Trimestre 2."
    )

    if not coverage.empty:
        total_efa = int(coverage["EFA_estudiantes"].sum())
        total_match = int(coverage["Emparejados"].sum())
        weighted_coverage = (
            total_match / total_efa * 100
            if total_efa > 0
            else np.nan
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Registros EFA con posible cruce",
            f"{total_efa:,}".replace(",", "."),
        )
        c2.metric(
            "Emparejados",
            f"{total_match:,}".replace(",", "."),
        )
        c3.metric(
            "Cobertura global",
            f"{weighted_coverage:.1f}%",
        )

    tests_available = [
        test for test in TESTS
        if test in set(summary["Prueba"].astype(str))
    ]
    test_tabs = st.tabs(tests_available)

    internal_order = [
        "Progreso limitado",
        "Emergente",
        "En aceleración",
        "Avanzado",
    ]
    trim_order = [
        "TRIMESTRE 1",
        "TRIMESTRE 2",
    ]
    grade_ranges = {
        "Matemáticas": list(range(3, 12)),
        "Lenguaje": list(range(3, 12)),
        "Ciencias naturales": list(range(5, 12)),
        "Ciencias sociales": list(range(5, 12)),
        "Inglés": list(range(9, 12)),
    }

    for tab, test in zip(test_tabs, tests_available):
        with tab:
            grades = grade_ranges.get(
                test,
                sorted(
                    summary.loc[
                        summary["Prueba"].astype(str) == test,
                        "Grado_num",
                    ]
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                ),
            )
            grade_labels = [f"{g}°" for g in grades]

            st.caption(
                "Los grados se mantienen fijos según la aplicabilidad de la "
                "prueba, hasta 11°."
            )

            st.markdown(
                "### Promedio EFA según nivel de logro interno · T1 → T2"
            )
            st.caption(
                "Para cada nivel interno se muestra el promedio exacto de EFA "
                "de los mismos estudiantes emparejados por correo. "
                "T1 y T2 aparecen juntos para comparar el recorrido."
            )

            rev = reverse_exact[
                (reverse_exact["Prueba"].astype(str) == test)
                & (
                    reverse_exact["Trimestre"]
                    .astype(str)
                    .isin(trim_order)
                )
            ].copy()

            if not rev.empty:
                rev["Nivel_interno"] = pd.Categorical(
                    rev["Nivel_interno"],
                    categories=internal_order,
                    ordered=True,
                )
                rev["Trimestre"] = pd.Categorical(
                    rev["Trimestre"],
                    categories=trim_order,
                    ordered=True,
                )

                template = pd.MultiIndex.from_product(
                    [grades, trim_order, internal_order],
                    names=[
                        "grado_num",
                        "Trimestre",
                        "Nivel_interno",
                    ],
                ).to_frame(index=False)

                rev = template.merge(
                    rev,
                    on=[
                        "grado_num",
                        "Trimestre",
                        "Nivel_interno",
                    ],
                    how="left",
                )
                rev["Estudiantes"] = pd.to_numeric(
                    rev["Estudiantes"],
                    errors="coerce",
                ).fillna(0).astype(int)
                rev["Etiqueta"] = rev["Promedio_EFA"].map(
                    lambda x: (
                        f"{x:.1f}%"
                        if pd.notna(x)
                        else "Sin dato"
                    )
                )

                fig_reverse = px.bar(
                    rev,
                    x="Nivel_interno",
                    y="Promedio_EFA",
                    color="Trimestre",
                    barmode="group",
                    facet_col="grado_num",
                    facet_col_wrap=3,
                    text="Etiqueta",
                    custom_data=["Estudiantes"],
                    category_orders={
                        "Nivel_interno": internal_order,
                        "Trimestre": trim_order,
                        "grado_num": grades,
                    },
                    title=(
                        f"{test} · promedio EFA por nivel interno · "
                        "Trimestre 1 vs Trimestre 2"
                    ),
                )
                fig_reverse.update_yaxes(
                    range=[0, 100],
                    title="Promedio EFA (%)",
                )
                fig_reverse.update_xaxes(title=None)
                fig_reverse.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "Nivel interno: %{x}<br>"
                        "Promedio EFA: %{y:.1f}%<br>"
                        "n emparejado: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                )
                fig_reverse.for_each_annotation(
                    lambda a: a.update(
                        text=a.text.replace(
                            "grado_num=", "Grado "
                        )
                        + "°"
                    )
                )
                fig_reverse.update_layout(
                    height=max(
                        620,
                        310 * int(np.ceil(len(grades) / 3)),
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Trimestre",
                )
                st.plotly_chart(
                    fig_reverse,
                    use_container_width=True,
                    key=f"internal_to_efa_t1_t2_{test}",
                )

                rev_table = rev[
                    rev["Promedio_EFA"].notna()
                ][
                    [
                        "grado_num",
                        "Trimestre",
                        "Nivel_interno",
                        "Estudiantes",
                        "Promedio_EFA",
                    ]
                ].copy()
                rev_table["Grado"] = rev_table[
                    "grado_num"
                ].map(lambda x: f"{int(x)}°")
                rev_table["Promedio EFA"] = rev_table[
                    "Promedio_EFA"
                ].map(lambda x: f"{x:.1f}%")
                rev_table["Trimestre"] = rev_table[
                    "Trimestre"
                ].astype(str).replace({
                    "TRIMESTRE 1": "T1",
                    "TRIMESTRE 2": "T2",
                })
                st.dataframe(
                    rev_table[
                        [
                            "Grado",
                            "Trimestre",
                            "Nivel_interno",
                            "Estudiantes",
                            "Promedio EFA",
                        ]
                    ].rename(
                        columns={
                            "Nivel_interno": "Nivel interno",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                "### Desempeño interno promedio según nivel EFA · T1 → T2"
            )
            st.caption(
                "Cada panel corresponde a un grado y muestra únicamente "
                "Trimestre 1 y Trimestre 2."
            )

            d = summary[
                (summary["Prueba"].astype(str) == test)
                & (
                    summary["Trimestre"]
                    .astype(str)
                    .isin(trim_order)
                )
            ].copy()

            template_d = pd.MultiIndex.from_product(
                [grades, trim_order, LEVEL_ORDER],
                names=[
                    "Grado_num",
                    "Trimestre",
                    "Nivel_EFA",
                ],
            ).to_frame(index=False)
            d = template_d.merge(
                d,
                on=[
                    "Grado_num",
                    "Trimestre",
                    "Nivel_EFA",
                ],
                how="left",
            )
            d["Nivel_EFA"] = pd.Categorical(
                d["Nivel_EFA"],
                categories=LEVEL_ORDER,
                ordered=True,
            )
            d["Trimestre"] = pd.Categorical(
                d["Trimestre"],
                categories=trim_order,
                ordered=True,
            )
            d["Estudiantes"] = pd.to_numeric(
                d["Estudiantes"],
                errors="coerce",
            ).fillna(0).astype(int)
            d["Etiqueta"] = d["Promedio_interno"].map(
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "Sin dato"
                )
            )

            fig = px.line(
                d,
                x="Nivel_EFA",
                y="Promedio_interno",
                color="Trimestre",
                facet_col="Grado_num",
                facet_col_wrap=3,
                markers=True,
                text="Etiqueta",
                custom_data=[
                    "Estudiantes",
                    "Promedio_EFA",
                ],
                category_orders={
                    "Nivel_EFA": LEVEL_ORDER,
                    "Trimestre": trim_order,
                    "Grado_num": grades,
                },
                title=(
                    f"{test} · desempeño interno por nivel EFA · "
                    "Trimestre 1 vs Trimestre 2"
                ),
            )
            fig.update_yaxes(
                range=[1, 4.15],
                title="Desempeño interno",
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=internal_order,
            )
            fig.update_xaxes(title=None)
            fig.update_traces(
                textposition="top center",
                cliponaxis=False,
                hovertemplate=(
                    "Nivel EFA: %{x}<br>"
                    "Desempeño interno: %{y:.2f}<br>"
                    "n: %{customdata[0]}<br>"
                    "Promedio EFA: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
            fig.for_each_annotation(
                lambda a: a.update(
                    text=a.text.replace(
                        "Grado_num=", "Grado "
                    )
                    + "°"
                )
            )
            fig.update_layout(
                height=max(
                    620,
                    310 * int(np.ceil(len(grades) / 3)),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Trimestre",
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"internal_efa_mean_t1_t2_{test}",
            )

            st.markdown(
                "### Distribución del nivel interno · T1 y T2"
            )
            st.caption(
                "Se presentan dos bloques separados para comparar el cambio "
                "entre Trimestre 1 y Trimestre 2 sin incorporar T3."
            )

            for trim in trim_order:
                st.markdown(
                    f"#### {trim.title()}"
                )
                dist = distribution[
                    (distribution["Prueba"].astype(str) == test)
                    & (
                        distribution["Trimestre"].astype(str)
                        == trim
                    )
                ].copy()

                template_dist = pd.MultiIndex.from_product(
                    [grades, LEVEL_ORDER, internal_order],
                    names=[
                        "Grado_num",
                        "Nivel_EFA",
                        "Interna_nivel",
                    ],
                ).to_frame(index=False)
                dist = template_dist.merge(
                    dist,
                    on=[
                        "Grado_num",
                        "Nivel_EFA",
                        "Interna_nivel",
                    ],
                    how="left",
                )
                dist["Porcentaje"] = pd.to_numeric(
                    dist["Porcentaje"],
                    errors="coerce",
                )
                dist["Nivel_EFA"] = pd.Categorical(
                    dist["Nivel_EFA"],
                    categories=LEVEL_ORDER,
                    ordered=True,
                )
                dist["Interna_nivel"] = pd.Categorical(
                    dist["Interna_nivel"],
                    categories=internal_order,
                    ordered=True,
                )

                fig_dist = px.bar(
                    dist,
                    x="Nivel_EFA",
                    y="Porcentaje",
                    color="Interna_nivel",
                    facet_col="Grado_num",
                    facet_col_wrap=3,
                    barmode="stack",
                    category_orders={
                        "Nivel_EFA": LEVEL_ORDER,
                        "Interna_nivel": internal_order,
                        "Grado_num": grades,
                    },
                    title=(
                        f"{test} · distribución interna por nivel EFA · "
                        f"{trim.title()}"
                    ),
                )
                fig_dist.update_yaxes(
                    range=[0, 100],
                    title="% estudiantes",
                )
                fig_dist.update_xaxes(title=None)
                fig_dist.for_each_annotation(
                    lambda a: a.update(
                        text=a.text.replace(
                            "Grado_num=", "Grado "
                        )
                        + "°"
                    )
                )
                fig_dist.update_layout(
                    height=max(
                        620,
                        310 * int(np.ceil(len(grades) / 3)),
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Desempeño interno",
                )
                st.plotly_chart(
                    fig_dist,
                    use_container_width=True,
                    key=(
                        f"internal_efa_distribution_"
                        f"{test}_{trim}"
                    ),
                )

            st.markdown("### Cobertura del cruce por grado")
            cov = coverage[
                coverage["Prueba"].astype(str) == test
            ].copy()
            grade_template = pd.DataFrame(
                {"Grado_num": grades}
            )
            cov = grade_template.merge(
                cov,
                on="Grado_num",
                how="left",
            )
            cov["Emparejados"] = pd.to_numeric(
                cov["Emparejados"],
                errors="coerce",
            ).fillna(0).astype(int)
            cov["EFA_estudiantes"] = pd.to_numeric(
                cov["EFA_estudiantes"],
                errors="coerce",
            ).fillna(0).astype(int)
            cov["Grado"] = cov["Grado_num"].map(
                lambda x: f"{int(x)}°"
            )
            cov["Etiqueta"] = cov["Cobertura_pct"].map(
                lambda x: (
                    f"{x:.1f}%"
                    if pd.notna(x)
                    else "Sin dato"
                )
            )

            fig_cov = px.bar(
                cov,
                x="Grado",
                y="Cobertura_pct",
                text="Etiqueta",
                custom_data=[
                    "Emparejados",
                    "EFA_estudiantes",
                ],
                category_orders={
                    "Grado": grade_labels,
                },
                title=f"{test} · cobertura de emparejamiento por correo",
            )
            fig_cov.update_yaxes(
                range=[0, 100],
                title="% emparejado",
            )
            fig_cov.update_xaxes(
                categoryorder="array",
                categoryarray=grade_labels,
            )
            fig_cov.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig_cov.update_layout(
                height=420,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_cov,
                use_container_width=True,
                key=f"internal_efa_coverage_{test}",
            )

            if not agreement.empty:
                agr = agreement[
                    (agreement["Prueba"].astype(str) == test)
                    & (
                        agreement["Trimestre"]
                        .astype(str)
                        .isin(trim_order)
                    )
                ].copy()

                if not agr.empty:
                    agr["Grado"] = agr["Grado_num"].map(
                        lambda x: f"{int(x)}°"
                    )
                    agr_long = agr.melt(
                        id_vars=["Grado", "Trimestre"],
                        value_vars=[
                            "Coincidencia_exacta_pct",
                            "Dentro_un_nivel_pct",
                        ],
                        var_name="Lectura",
                        value_name="Porcentaje",
                    )
                    agr_long["Lectura"] = agr_long[
                        "Lectura"
                    ].replace({
                        "Coincidencia_exacta_pct": "Mismo nivel",
                        "Dentro_un_nivel_pct": "Máximo ±1 nivel",
                    })

                    st.markdown(
                        "### Cercanía entre ambas escalas · T1 y T2"
                    )
                    fig_agr = px.bar(
                        agr_long,
                        x="Grado",
                        y="Porcentaje",
                        color="Lectura",
                        facet_col="Trimestre",
                        barmode="group",
                        text=agr_long["Porcentaje"].map(
                            lambda x: f"{x:.1f}%"
                        ),
                        category_orders={
                            "Grado": grade_labels,
                            "Trimestre": trim_order,
                        },
                        title=(
                            f"{test} · cercanía entre nivel interno y EFA · "
                            "Trimestre 1 vs Trimestre 2"
                        ),
                    )
                    fig_agr.update_yaxes(
                        range=[0, 100],
                        title="% estudiantes",
                    )
                    fig_agr.update_traces(
                        textposition="outside",
                        cliponaxis=False,
                    )
                    fig_agr.for_each_annotation(
                        lambda a: a.update(
                            text=a.text.replace(
                                "Trimestre=", ""
                            ).title()
                        )
                    )
                    fig_agr.update_layout(
                        height=480,
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend_title_text="",
                    )
                    st.plotly_chart(
                        fig_agr,
                        use_container_width=True,
                        key=f"internal_efa_agreement_t1_t2_{test}",
                    )


def _story_level_from_pct(value) -> str:
    if pd.isna(value):
        return "Sin dato"
    value = float(value)
    if value <= 25:
        return "Nivel 1"
    if value <= 50:
        return "Nivel 2"
    if value <= 75:
        return "Nivel 3"
    return "Nivel 4"


def _render_internal_story(
    internal_summary: pd.DataFrame,
    grade: int,
    test: str,
    level: str,
    key: str,
):
    if internal_summary is None or internal_summary.empty:
        return

    d = internal_summary[
        (internal_summary["Grado_num"] == grade)
        & (internal_summary["Prueba"].astype(str) == test)
        & (internal_summary["Nivel_EFA"].astype(str) == level)
    ].copy()

    if d.empty:
        return

    trim_order = ["TRIMESTRE 1", "TRIMESTRE 2"]
    d = d[
        d["Trimestre"].astype(str).isin(trim_order)
    ].copy()
    d["Trimestre"] = pd.Categorical(
        d["Trimestre"],
        categories=trim_order,
        ordered=True,
    )
    d = d.sort_values("Trimestre")

    st.markdown("### Evaluación interna · trayectoria durante el año")
    fig = px.line(
        d,
        x="Trimestre",
        y="Promedio_interno",
        markers=True,
        text=d["Promedio_interno"].map(
            lambda x: f"{x:.2f}"
        ),
        custom_data=["Estudiantes"],
        category_orders={"Trimestre": trim_order},
        title=(
            f"Desempeño interno de estudiantes en {level} de {test}"
        ),
    )
    fig.update_yaxes(
        range=[1, 4.15],
        title="Desempeño interno",
        tickmode="array",
        tickvals=[1, 2, 3, 4],
        ticktext=[
            "Progreso limitado",
            "Emergente",
            "En aceleración",
            "Avanzado",
        ],
    )
    fig.update_xaxes(title=None)
    fig.update_traces(
        textposition="top center",
        cliponaxis=False,
        hovertemplate=(
            "Trimestre: %{x}<br>"
            "Desempeño interno: %{y:.2f}<br>"
            "n: %{customdata[0]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


def render_student_story(
    profiles: pd.DataFrame,
    profiles_3_9: pd.DataFrame,
    internal_summary: pd.DataFrame,
):
    st.markdown("## Historia de un perfil representativo")
    st.caption(
        "Selecciona grado, nivel EFA y prueba EFA para reconstruir el "
        "recorrido disponible en las demás evaluaciones."
    )

    if profiles.empty and profiles_3_9.empty:
        st.warning(
            "No está disponible la base de perfiles representativos."
        )
        return

    st.info(
        "La vista trabaja con perfiles agregados de grupos de al menos "
        "5 estudiantes. De 3° a 9° se muestran EFA, Innovamat y LATAM "
        "cuando existe emparejamiento suficiente, sin Improve. "
        "En 10° y 11° se conserva además la trayectoria de Improve."
    )

    grades = sorted(
        set(
            profiles["grado_num"].dropna().astype(int).tolist()
            if not profiles.empty
            else []
        )
        | set(
            profiles_3_9["grado_num"].dropna().astype(int).tolist()
            if not profiles_3_9.empty
            else []
        )
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_grade = st.selectbox(
            "Grado",
            grades,
            format_func=lambda g: f"{g}°",
            key="story_grade",
        )

    if selected_grade <= 9:
        grade_data = profiles_3_9[
            profiles_3_9["grado_num"] == selected_grade
        ].copy()

        available_levels = [
            level
            for level in LEVEL_ORDER
            if level in set(
                grade_data["Nivel_EFA_filtro"].astype(str)
            )
        ]

        with c2:
            selected_level = st.selectbox(
                "Nivel en Evaluar para Avanzar",
                available_levels,
                key="story_level",
            )

        level_data = grade_data[
            grade_data["Nivel_EFA_filtro"].astype(str)
            == selected_level
        ].copy()

        available_tests = [
            test
            for test in IMPROVE_EFA_TESTS
            if test in set(
                level_data["Prueba_filtro"].astype(str)
            )
        ]

        with c3:
            selected_test = st.selectbox(
                "Prueba de Evaluar para Avanzar",
                available_tests,
                key="story_test",
            )

        d = level_data[
            level_data["Prueba_filtro"].astype(str)
            == selected_test
        ].copy()

        if d.empty:
            st.warning(
                "No hay un grupo representativo para esta combinación."
            )
            return

        group_n = int(d["Grupo_EFA_n"].max())
        efa_selected = d[
            (d["Fuente"].astype(str) == "Evaluar para Avanzar")
            & (d["Resultado"].astype(str) == selected_test)
        ].copy()
        anchor_score = (
            float(efa_selected.iloc[0]["Valor"])
            if not efa_selected.empty
            else np.nan
        )

        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Estudiantes en el grupo",
            f"{group_n}",
        )
        k2.metric(
            f"EFA · {selected_test}",
            (
                f"{anchor_score:.1f}%"
                if np.isfinite(anchor_score)
                else "—"
            ),
            delta=selected_level,
            delta_color="off",
        )
        sources = [
            src
            for src in ["Evaluar para Avanzar", "Innovamat", "LATAM"]
            if src in set(d["Fuente"].astype(str))
        ]
        k3.metric(
            "Fuentes disponibles",
            f"{len(sources)}",
            help=", ".join(sources),
        )

        st.markdown("### 1. Perfil en Evaluar para Avanzar")
        efa = d[
            d["Fuente"].astype(str) == "Evaluar para Avanzar"
        ].copy()
        if not efa.empty:
            efa["Selección"] = np.where(
                efa["Resultado"].astype(str) == selected_test,
                "Prueba seleccionada",
                "Otras pruebas",
            )
            efa["Nivel"] = efa["Valor"].map(_story_level_from_pct)

            fig_efa = px.bar(
                efa,
                x="Resultado",
                y="Valor",
                color="Selección",
                text=efa["Valor"].map(lambda x: f"{x:.1f}%"),
                custom_data=["n", "Nivel"],
                category_orders={
                    "Resultado": IMPROVE_EFA_TESTS,
                },
                title=(
                    f"Perfil EFA representativo · {selected_grade}° · "
                    f"{selected_level} en {selected_test}"
                ),
            )
            for y in [25, 50, 75]:
                fig_efa.add_hline(
                    y=y,
                    line_dash="dot",
                    opacity=0.35,
                )
            fig_efa.update_yaxes(
                range=[0, 100],
                title="Puntaje EFA (%)",
            )
            fig_efa.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "Prueba: %{x}<br>"
                    "Puntaje EFA: %{y:.1f}%<br>"
                    "n: %{customdata[0]}<br>"
                    "Nivel: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
            fig_efa.update_layout(
                height=470,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
            )
            st.plotly_chart(
                fig_efa,
                use_container_width=True,
                key="story_efa_profile_3_9",
            )

        _render_internal_story(
            internal_summary,
            selected_grade,
            selected_test,
            selected_level,
            "story_internal_3_9",
        )

        st.markdown("### 2. Resultados adicionales disponibles")

        innovamat = d[
            d["Fuente"].astype(str) == "Innovamat"
        ].copy()
        if not innovamat.empty:
            st.markdown("#### Innovamat · perfil matemático")
            innovamat_order = [
                "Promedio general",
                "Numeración",
                "Relaciones y cambio",
                "Espacio y forma",
                "Medida",
                "Estadística y probabilidad",
            ]
            innovamat["Resultado"] = pd.Categorical(
                innovamat["Resultado"],
                categories=innovamat_order,
                ordered=True,
            )
            innovamat = innovamat.sort_values("Resultado")

            fig_inv = px.bar(
                innovamat,
                x="Resultado",
                y="Valor",
                text=innovamat["Valor"].map(
                    lambda x: f"{x:.1f}"
                ),
                custom_data=["n"],
                category_orders={
                    "Resultado": innovamat_order,
                },
                title=(
                    "Desempeño representativo en la actividad "
                    "matemática adicional"
                ),
            )
            fig_inv.update_yaxes(
                range=[0, 100],
                title="Puntaje equivalente (0–100)",
            )
            fig_inv.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "Dimensión: %{x}<br>"
                    "Resultado: %{y:.1f}/100<br>"
                    "n: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )
            fig_inv.update_layout(
                height=480,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_inv,
                use_container_width=True,
                key="story_innovamat_3_9",
            )

        latam = d[
            d["Fuente"].astype(str) == "LATAM"
        ].copy()
        if not latam.empty:
            st.markdown("#### LATAM · nivel de logro")
            latam_order = [
                "Matemáticas",
                "Lenguaje",
            ]
            latam["Resultado"] = pd.Categorical(
                latam["Resultado"],
                categories=latam_order,
                ordered=True,
            )
            latam = latam.sort_values("Resultado")

            fig_latam = px.bar(
                latam,
                x="Resultado",
                y="Valor",
                text=latam["Etiqueta"],
                custom_data=["n", "Extra"],
                category_orders={
                    "Resultado": latam_order,
                },
                title="Nivel de logro LATAM predominante",
            )
            fig_latam.update_yaxes(
                range=[0, 4.5],
                title="Nivel LATAM (1–4)",
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=[
                    "Inicio",
                    "Básico",
                    "Satisfactorio",
                    "Avanzado",
                ],
            )
            fig_latam.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "Área: %{x}<br>"
                    "Nivel predominante: %{text}<br>"
                    "n: %{customdata[0]}<br>"
                    "Proporción en ese nivel: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
            fig_latam.update_layout(
                height=420,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_latam,
                use_container_width=True,
                key="story_latam_3_9",
            )

        if innovamat.empty and latam.empty:
            st.caption(
                "Para esta combinación no hay otra fuente con al menos "
                "5 estudiantes emparejados además de EFA."
            )

        st.markdown("### 3. La historia en una frase")
        extra_bits = []
        if not innovamat.empty:
            inv_general = innovamat[
                innovamat["Resultado"].astype(str)
                == "Promedio general"
            ]
            if not inv_general.empty:
                extra_bits.append(
                    "en Innovamat el promedio general representativo es "
                    f"**{float(inv_general.iloc[0]['Valor']):.1f}/100**"
                )
        if not latam.empty:
            latam_txt = ", ".join(
                f"{str(r['Resultado'])}: **{r['Etiqueta']}**"
                for _, r in latam.iterrows()
            )
            extra_bits.append(
                f"en LATAM predominan {latam_txt}"
            )

        story_extra = (
            "; ".join(extra_bits)
            if extra_bits
            else "no hay otra evaluación con muestra suficiente"
        )
        st.success(
            f"En {selected_grade}°, este perfil representa a estudiantes "
            f"ubicados en **{selected_level} de {selected_test}** en EFA"
            + (
                f", con una mediana de **{anchor_score:.1f}%**. "
                if np.isfinite(anchor_score)
                else ". "
            )
            + story_extra
            + "."
        )

        st.caption(
            "De 3° a 9° no se incorpora Improve. Cada resultado adicional "
            "solo aparece si hay al menos 5 estudiantes emparejados dentro "
            "del grupo seleccionado."
        )
        return

    # 10° y 11° mantienen la historia longitudinal con Improve.
    grade_data = profiles[
        profiles["grado_num"] == selected_grade
    ].copy()

    available_levels = [
        level
        for level in LEVEL_ORDER
        if level in set(
            grade_data["Nivel_EFA_filtro"].astype(str)
        )
    ]

    with c2:
        selected_level = st.selectbox(
            "Nivel en Evaluar para Avanzar",
            available_levels,
            key="story_level",
        )

    level_data = grade_data[
        grade_data["Nivel_EFA_filtro"].astype(str)
        == selected_level
    ].copy()

    available_tests = [
        test
        for test in IMPROVE_EFA_TESTS
        if test in set(
            level_data["Prueba_filtro"].astype(str)
        )
    ]

    with c3:
        selected_test = st.selectbox(
            "Prueba de Evaluar para Avanzar",
            available_tests,
            key="story_test",
        )

    d = level_data[
        level_data["Prueba_filtro"].astype(str)
        == selected_test
    ].copy()

    if d.empty:
        st.warning(
            "No hay un grupo con recorrido completo para esta combinación."
        )
        return

    row = d.iloc[0]
    group_n = int(row["Candidatos_completos"])
    anchor_score = float(row["EFA_seleccionada_mediana"])

    final_ev = 2 if selected_grade == 10 else 3
    final_label = f"Evaluación {final_ev}"

    global_initial = float(
        row["Improve_EV1_Puntaje global_mediana"]
    )
    global_final = float(
        row[f"Improve_EV{final_ev}_Puntaje global_mediana"]
    )
    global_change = global_final - global_initial

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Grupo con recorrido completo",
        f"{group_n}",
        help=(
            "Número de estudiantes usados para construir este perfil "
            "representativo."
        ),
    )
    k2.metric(
        f"EFA · {selected_test}",
        f"{anchor_score:.1f}%",
        delta=selected_level,
        delta_color="off",
    )
    k3.metric(
        "Improve global · inicio",
        f"{global_initial:.1f}",
    )
    k4.metric(
        f"Improve global · {final_label.lower()}",
        f"{global_final:.1f}",
        delta=f"{global_change:+.1f}",
    )

    st.markdown("### 1. El perfil en Evaluar para Avanzar")

    efa_rows = []
    for test in IMPROVE_EFA_TESTS:
        value = row.get(f"EFA_{test}_mediana", np.nan)
        if pd.isna(value):
            continue
        efa_rows.append({
            "Prueba": test,
            "Puntaje EFA": float(value),
            "Nivel": _story_level_from_pct(value),
            "Selección": (
                "Prueba seleccionada"
                if test == selected_test
                else "Otras pruebas"
            ),
        })

    efa_profile = pd.DataFrame(efa_rows)

    if not efa_profile.empty:
        fig_efa = px.bar(
            efa_profile,
            x="Prueba",
            y="Puntaje EFA",
            color="Selección",
            text=efa_profile["Puntaje EFA"].map(
                lambda x: f"{x:.1f}%"
            ),
            custom_data=["Nivel"],
            title=(
                f"Perfil EFA representativo · {selected_grade}° · "
                f"{selected_level} en {selected_test}"
            ),
        )
        for y in [25, 50, 75]:
            fig_efa.add_hline(
                y=y,
                line_dash="dot",
                opacity=0.35,
            )
        fig_efa.update_yaxes(
            range=[0, 100],
            title="Puntaje EFA (%)",
        )
        fig_efa.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Prueba: %{x}<br>"
                "Puntaje EFA: %{y:.1f}%<br>"
                "Nivel: %{customdata[0]}"
                "<extra></extra>"
            ),
        )
        fig_efa.update_layout(
            height=480,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(
            fig_efa,
            use_container_width=True,
            key="story_efa_profile",
        )

    _render_internal_story(
        internal_summary,
        selected_grade,
        selected_test,
        selected_level,
        "story_internal_10_11",
    )

    st.markdown("### 2. La trayectoria global en Improve")

    moments = [1, 2] if selected_grade == 10 else [1, 2, 3]
    global_rows = []
    for moment in moments:
        value = row.get(
            f"Improve_EV{moment}_Puntaje global_mediana",
            np.nan,
        )
        if pd.isna(value):
            continue
        global_rows.append({
            "Momento": f"Evaluación {moment}",
            "Orden": moment,
            "Puntaje global": float(value),
        })

    global_traj = pd.DataFrame(global_rows)
    if not global_traj.empty:
        fig_global = px.line(
            global_traj.sort_values("Orden"),
            x="Momento",
            y="Puntaje global",
            markers=True,
            text=global_traj["Puntaje global"].map(
                lambda x: f"{x:.1f}"
            ),
            category_orders={
                "Momento": [
                    f"Evaluación {m}" for m in moments
                ]
            },
            title="Cómo cambia el puntaje global de Improve",
        )
        fig_global.update_traces(
            textposition="top center",
            cliponaxis=False,
        )
        fig_global.update_yaxes(
            rangemode="tozero",
            title="Puntaje global Improve",
        )
        fig_global.update_layout(
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_global,
            use_container_width=True,
            key="story_improve_global",
        )

    st.markdown("### 3. Qué ocurre en cada prueba de Improve")

    improve_map = {
        "Matemáticas": "Puntaje Matemáticas",
        "Lenguaje": "Puntaje Lectura crítica",
        "Ciencias naturales": "Puntaje Ciencias naturales",
        "Ciencias sociales": "Puntaje Sociales y ciudadanas",
        "Inglés": "Puntaje Inglés",
    }

    subject_rows = []
    for test, raw_name in improve_map.items():
        for moment in moments:
            value = row.get(
                f"Improve_EV{moment}_{raw_name}_mediana",
                np.nan,
            )
            if pd.isna(value):
                continue
            subject_rows.append({
                "Prueba": test,
                "Momento": f"Evaluación {moment}",
                "Orden": moment,
                "Promedio Improve": float(value),
            })

    subject_traj = pd.DataFrame(subject_rows)

    if not subject_traj.empty:
        fig_subject = px.line(
            subject_traj.sort_values(["Prueba", "Orden"]),
            x="Momento",
            y="Promedio Improve",
            color="Prueba",
            markers=True,
            category_orders={
                "Momento": [
                    f"Evaluación {m}" for m in moments
                ],
                "Prueba": IMPROVE_EFA_TESTS,
            },
            title="Recorrido por las cinco pruebas de Improve",
        )
        fig_subject.update_yaxes(
            range=[0, 100],
            title="Puntaje Improve",
        )
        fig_subject.update_xaxes(title=None)
        fig_subject.update_layout(
            height=540,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Prueba",
        )
        st.plotly_chart(
            fig_subject,
            use_container_width=True,
            key="story_improve_subjects",
        )

        table_rows = []
        for test, raw_name in improve_map.items():
            efa_value = row.get(
                f"EFA_{test}_mediana",
                np.nan,
            )
            initial = row.get(
                f"Improve_EV1_{raw_name}_mediana",
                np.nan,
            )
            final = row.get(
                f"Improve_EV{final_ev}_{raw_name}_mediana",
                np.nan,
            )
            if pd.isna(initial) or pd.isna(final):
                continue
            table_rows.append({
                "Prueba": test,
                "EFA": float(efa_value)
                if not pd.isna(efa_value)
                else np.nan,
                "Nivel EFA": _story_level_from_pct(efa_value),
                "Improve inicial": float(initial),
                "Improve final": float(final),
                "Cambio": float(final) - float(initial),
            })

        story_table = pd.DataFrame(table_rows)

        if not story_table.empty:
            display = story_table.copy()
            display["EFA"] = display["EFA"].map(
                lambda x: f"{x:.1f}%"
            )
            display["Improve inicial"] = display[
                "Improve inicial"
            ].map(lambda x: f"{x:.1f}")
            display["Improve final"] = display[
                "Improve final"
            ].map(lambda x: f"{x:.1f}")
            display["Cambio"] = display["Cambio"].map(
                lambda x: f"{x:+.1f}"
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

            best = story_table.sort_values(
                "Cambio",
                ascending=False,
            ).iloc[0]
            worst = story_table.sort_values(
                "Cambio",
                ascending=True,
            ).iloc[0]

            st.markdown("### 4. La historia en una frase")
            direction = (
                "aumenta"
                if global_change > 0
                else "disminuye"
                if global_change < 0
                else "se mantiene"
            )
            st.success(
                f"Este perfil representativo parte de **{selected_level} en "
                f"{selected_test}** ({anchor_score:.1f}% en EFA). "
                f"En Improve, su puntaje global {direction} de "
                f"**{global_initial:.1f}** a **{global_final:.1f}** "
                f"({global_change:+.1f} puntos). La mayor mejora por prueba "
                f"se observa en **{best['Prueba']}** "
                f"({best['Cambio']:+.1f}), mientras que el cambio más bajo "
                f"se presenta en **{worst['Prueba']}** "
                f"({worst['Cambio']:+.1f})."
            )

    st.caption(
        "El perfil representa el comportamiento mediano de un grupo con "
        "recorrido completo; no corresponde al expediente de una persona "
        "identificable."
    )


st.title("📊 Estado de llegada 2026")
st.caption(
    "Base precargada actualizada al 19 de octubre de 2026 · "
    "Análisis general, antigüedad, LATAM, actividad matemática, evolución, relación EFA–Improve, evaluación interna e historia longitudinal."
)

uploaded = st.file_uploader(
    "Cargar Excel completo (opcional)",
    type=["xlsx", "xls"],
    help=(
        "La base precargada permite ambas ventanas. "
        "El Excel completo habilita también los análisis individuales."
    ),
)

aggregated = uploaded is None

if uploaded is not None:
    try:
        raw = read_excel(uploaded.getvalue())
        (
            attempts,
            dimensions,
            age_attempts,
            age_dimensions,
        ) = process_raw(raw)

        levels_public = pd.DataFrame()
        top10_public = pd.DataFrame()
        age_levels_public = pd.DataFrame()
        age_top10_public = pd.DataFrame()
    except Exception as exc:
        st.error(f"No fue posible procesar el Excel: {exc}")
        st.stop()
else:
    (
        attempts,
        levels_public,
        dimensions,
        top10_public,
        age_attempts,
        age_levels_public,
        age_dimensions,
        age_top10_public,
    ) = read_default_data()

    if attempts.empty:
        st.info("Carga el Excel para visualizar los resultados.")
        st.stop()

latam_efa_data, latam_efa_summary = read_latam_efa_data()
math_activity_stats, math_activity_bins, math_activity_summary = (
    read_math_activity_data()
)
(
    evolution_summary,
    evolution_trajectory,
    evolution_availability,
    evolution_observed,
) = read_evolution_10_11_data()
(
    improve_efa_levels,
    improve_efa_quality,
    improve_efa_growth,
) = read_improve_efa_data()
(
    internal_efa_summary,
    internal_efa_distribution,
    internal_efa_coverage,
    internal_efa_agreement,
    internal_efa_reverse,
) = read_internal_efa_data()
student_story_profiles = read_student_story_profiles()
student_story_3_9 = read_student_story_3_9()

(
    view_general,
    view_age,
    view_latam,
    view_math_activity,
    view_evolution,
    view_improve_efa,
    view_internal_efa,
    view_story,
) = st.tabs(
    [
        "Análisis general",
        "Análisis por antigüedad ≤1 año",
        "LATAM vs Evaluar para Avanzar",
        "Actividad matemática vs EFA",
        "Evolución 10°–11°",
        "EFA vs Improve",
        "Interna vs EFA",
        "Historia longitudinal",
    ]
)

with view_general:
    st.caption(
        "Vista general de resultados por prueba, sede, grado, niveles, "
        "dimensiones y colegio de origen."
    )
    general_tabs = st.tabs(TESTS)
    for tab, test_name in zip(general_tabs, TESTS):
        with tab:
            render_test_page(
                attempts=attempts,
                dimensions=dimensions,
                levels_public=levels_public,
                top10_public=top10_public,
                test_name=test_name,
                aggregated=aggregated,
                view_key="general",
                cohort_caption="Todos los estudiantes",
            )

with view_age:
    st.info(
        "Esta ventana analiza exclusivamente estudiantes de COT, MOS, TUN y ZIPA "
        "con 1 año o menos de antigüedad. Todas las gráficas, distribuciones y "
        "comparaciones se recalculan dentro de esta cohorte."
    )

    if age_attempts.empty:
        st.warning("No hay datos disponibles para la cohorte de antigüedad.")
    else:
        age_tabs = st.tabs(TESTS)
        for tab, test_name in zip(age_tabs, TESTS):
            with tab:
                render_test_page(
                    attempts=age_attempts,
                    dimensions=age_dimensions,
                    levels_public=age_levels_public,
                    top10_public=age_top10_public,
                    test_name=test_name,
                    aggregated=aggregated,
                    view_key="antiguedad",
                    cohort_caption=(
                        "COT, MOS, TUN y ZIPA · 1 año o menos de antigüedad"
                    ),
                )


with view_latam:
    st.info(
        "Esta ventana cruza únicamente Matemáticas y Lenguaje. "
        "LATAM se empareja por correo mediante la base de usuarios y luego "
        "con Evaluar para Avanzar por identificador de estudiante. "
        "NIZA está excluida, igual que en el resto del dashboard."
    )

    if latam_efa_data.empty:
        st.warning(
            "No está disponible el resumen de emparejamiento LATAM–EFA."
        )
    else:
        latam_math, latam_language = st.tabs(
            ["Matemáticas", "Lenguaje"]
        )
        with latam_math:
            render_latam_efa_area(
                latam_efa_data,
                latam_efa_summary,
                "Matemáticas",
            )
        with latam_language:
            render_latam_efa_area(
                latam_efa_data,
                latam_efa_summary,
                "Lenguaje",
            )


with view_math_activity:
    render_math_activity_vs_efa(
        math_activity_stats,
        math_activity_bins,
        math_activity_summary,
    )



with view_evolution:
    render_evolution_10_11(
        evolution_summary,
        evolution_trajectory,
        evolution_availability,
        evolution_observed,
    )



with view_improve_efa:
    render_improve_efa_relation(
        improve_efa_levels,
        improve_efa_quality,
        improve_efa_growth,
    )



with view_internal_efa:
    render_internal_vs_efa(
        internal_efa_summary,
        internal_efa_distribution,
        internal_efa_coverage,
        internal_efa_agreement,
        internal_efa_reverse,
    )


with view_story:
    render_student_story(
        student_story_profiles,
        student_story_3_9,
        internal_efa_summary,
    )
