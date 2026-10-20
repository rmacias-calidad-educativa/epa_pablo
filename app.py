
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

    for df in [summary, trajectory, availability]:
        if not df.empty and "Grado_num" in df.columns:
            df["Grado_num"] = pd.to_numeric(
                df["Grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["Grado_num"], inplace=True)
            df["Grado_num"] = df["Grado_num"].astype(int)

    return summary, trajectory, availability


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


def render_evolution_10_11(
    summary: pd.DataFrame,
    trajectory: pd.DataFrame,
    availability: pd.DataFrame,
):
    st.markdown("## Evolución de resultados · 10° y 11°")
    st.caption(
        "Comparación pareada entre la primera y la última evaluación "
        "disponible de cada estudiante."
    )

    if summary.empty:
        st.warning("No está disponible el resumen de evolución.")
        return

    st.info(
        "La comparación utiliza únicamente estudiantes con primera y última "
        "evaluación. En 10° se compara Inicial vs Final; en 11° se compara "
        "Inicial vs Final. La evaluación intermedia de 11° se muestra como "
        "trayectoria complementaria. Cota no tiene aplicación final de 10°, "
        "por lo que no entra en la comparación longitudinal."
    )

    available_grades = sorted(
        [int(g) for g in summary["Grado_num"].dropna().unique()]
    )
    available_sites = sorted(
        summary["Sede"].dropna().astype(str).unique().tolist()
    )

    f1, f2, f3 = st.columns([1.1, 1.2, 1.7])
    with f1:
        selected_metric = st.selectbox(
            "Resultado",
            EVOLUTION_METRICS,
            index=0,
            key="evolution_metric",
        )
    with f2:
        selected_grades = st.multiselect(
            "Grados",
            available_grades,
            default=available_grades,
            format_func=lambda g: f"{g}°",
            key="evolution_grades",
        )
    with f3:
        selected_sites = st.multiselect(
            "Sedes",
            available_sites,
            default=available_sites,
            key="evolution_sites",
        )

    if not selected_grades or not selected_sites:
        st.warning("Selecciona al menos un grado y una sede.")
        return

    d = _evolution_filter(
        summary,
        metric=selected_metric,
        selected_grades=selected_grades,
        selected_sites=selected_sites,
    )
    if d.empty:
        st.warning("No hay estudiantes pareados para los filtros seleccionados.")
        return

    overall = _evolution_combine(d)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Estudiantes pareados",
        f"{overall['n']:,}".replace(",", "."),
    )
    k2.metric(
        "Promedio inicial",
        f"{overall['initial']:.1f}",
    )
    k3.metric(
        "Promedio final",
        f"{overall['final']:.1f}",
    )
    k4.metric(
        "Cambio promedio",
        f"{overall['change']:+.1f}",
        help="Promedio final menos promedio inicial.",
    )
    k5.metric(
        "% que mejora",
        f"{overall['improved']:.1f}%",
    )

    if len(selected_grades) > 1:
        st.caption(
            "El resumen superior combina los grados seleccionados. Para comparar "
            "sedes de forma justa, revisa el desglose por grado: 11° solo tiene "
            "trayectoria completa en Mosquera."
        )

    by_grade = _evolution_grouped(d, ["Grado_num"])
    if not by_grade.empty:
        by_grade["Grado"] = by_grade["Grado_num"].map(
            lambda g: f"{int(g)}°"
        )
        grade_long = pd.concat(
            [
                by_grade[["Grado", "Promedio inicial"]]
                .rename(columns={"Promedio inicial": "Promedio"})
                .assign(Momento="Inicial"),
                by_grade[["Grado", "Promedio final"]]
                .rename(columns={"Promedio final": "Promedio"})
                .assign(Momento="Final"),
            ],
            ignore_index=True,
        )
        fig_grade = px.bar(
            grade_long,
            x="Grado",
            y="Promedio",
            color="Momento",
            barmode="group",
            text=grade_long["Promedio"].map(lambda x: f"{x:.1f}"),
            title="Promedio inicial vs final por grado",
        )
        fig_grade.update_traces(
            textposition="outside",
            cliponaxis=False,
        )
        fig_grade.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Momento",
        )
        st.plotly_chart(
            fig_grade,
            use_container_width=True,
            key="evolution_initial_final_by_grade",
        )

    st.divider()
    st.markdown("### ¿En qué sede se presenta mayor avance?")

    by_site_grade = _evolution_grouped(
        d,
        ["Grado_num", "Sede"],
    )
    if not by_site_grade.empty:
        by_site_grade["Grado"] = by_site_grade["Grado_num"].map(
            lambda g: f"{int(g)}°"
        )

        fig_change = px.bar(
            by_site_grade.sort_values(
                ["Grado_num", "Cambio promedio"],
                ascending=[True, False],
            ),
            x="Sede",
            y="Cambio promedio",
            color="Grado",
            barmode="group",
            text=by_site_grade["Cambio promedio"].map(
                lambda x: f"{x:+.1f}"
            ),
            custom_data=[
                "N",
                "Promedio inicial",
                "Promedio final",
                "% mejora",
            ],
            labels={"Cambio promedio": "Cambio promedio"},
            title="Avance promedio por sede y grado",
        )
        fig_change.add_hline(y=0, line_dash="dot")
        fig_change.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Cambio promedio: %{y:+.1f}<br>"
                "n: %{customdata[0]}<br>"
                "Inicial: %{customdata[1]:.1f}<br>"
                "Final: %{customdata[2]:.1f}<br>"
                "Mejoran: %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
        )
        fig_change.update_layout(
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Grado",
        )
        st.plotly_chart(
            fig_change,
            use_container_width=True,
            key="evolution_change_by_site_grade",
        )

        ranking = by_site_grade.copy()
        ranking["Promedio inicial"] = ranking[
            "Promedio inicial"
        ].map(lambda x: f"{x:.1f}")
        ranking["Promedio final"] = ranking[
            "Promedio final"
        ].map(lambda x: f"{x:.1f}")
        ranking["Cambio promedio"] = ranking[
            "Cambio promedio"
        ].map(lambda x: f"{x:+.1f}")
        ranking["% mejora"] = ranking[
            "% mejora"
        ].map(lambda x: f"{x:.1f}%")
        ranking["% baja"] = ranking[
            "% baja"
        ].map(lambda x: f"{x:.1f}%")
        st.dataframe(
            ranking[
                [
                    "Grado",
                    "Sede",
                    "N",
                    "Promedio inicial",
                    "Promedio final",
                    "Cambio promedio",
                    "% mejora",
                    "% baja",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        for grade in sorted(by_site_grade["Grado_num"].unique()):
            rg = by_site_grade[
                by_site_grade["Grado_num"] == grade
            ].sort_values("Cambio promedio", ascending=False)
            if len(rg) > 1:
                best = rg.iloc[0]
                st.success(
                    f"En {int(grade)}°, la sede con mayor avance promedio "
                    f"en {selected_metric} es **{best['Sede']}**, con "
                    f"{best['Cambio promedio']:+.1f} puntos."
                )
            elif len(rg) == 1:
                only = rg.iloc[0]
                st.caption(
                    f"En {int(grade)}° solo hay una sede con comparación "
                    f"completa: {only['Sede']} "
                    f"({only['Cambio promedio']:+.1f} puntos)."
                )

    st.divider()
    st.markdown("### Evolución por área de evaluación")
    area_rows = []
    for metric in EVOLUTION_METRICS:
        dm = _evolution_filter(
            summary,
            metric=metric,
            selected_grades=selected_grades,
            selected_sites=selected_sites,
        )
        info = _evolution_combine(dm)
        if info["n"] == 0:
            continue
        area_rows.append({
            "Resultado": metric.replace("Puntaje ", ""),
            "n": info["n"],
            "Promedio inicial": info["initial"],
            "Promedio final": info["final"],
            "Cambio promedio": info["change"],
            "% mejora": info["improved"],
        })

    area_table = pd.DataFrame(area_rows)
    if not area_table.empty:
        area_display = area_table.copy()
        for col in [
            "Promedio inicial",
            "Promedio final",
        ]:
            area_display[col] = area_display[col].map(
                lambda x: f"{x:.1f}"
            )
        area_display["Cambio promedio"] = area_display[
            "Cambio promedio"
        ].map(lambda x: f"{x:+.1f}")
        area_display["% mejora"] = area_display[
            "% mejora"
        ].map(lambda x: f"{x:.1f}%")
        st.dataframe(
            area_display,
            use_container_width=True,
            hide_index=True,
        )

        fig_area = px.bar(
            area_table,
            x="Resultado",
            y="Cambio promedio",
            text=area_table["Cambio promedio"].map(
                lambda x: f"{x:+.1f}"
            ),
            title="Cambio promedio entre primera y última evaluación",
        )
        fig_area.add_hline(y=0, line_dash="dot")
        fig_area.update_traces(
            textposition="outside",
            cliponaxis=False,
        )
        fig_area.update_layout(
            height=470,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_area,
            use_container_width=True,
            key="evolution_change_by_subject",
        )

    if 11 in selected_grades and not trajectory.empty:
        st.divider()
        st.markdown("### Trayectoria de 11° · Inicial → Intermedia → Final")
        t = _evolution_filter(
            trajectory,
            metric=selected_metric,
            selected_grades=[11],
            selected_sites=selected_sites,
        )
        if not t.empty:
            traj = (
                t.groupby(
                    ["Sede", "Momento", "Orden"],
                    as_index=False,
                )
                .agg(
                    N=("N", "sum"),
                    Sum_Puntaje=("Sum_Puntaje", "sum"),
                )
            )
            traj["Promedio"] = np.where(
                traj["N"] > 0,
                traj["Sum_Puntaje"] / traj["N"],
                np.nan,
            )
            traj = traj.sort_values(["Sede", "Orden"])

            fig_traj = px.line(
                traj,
                x="Momento",
                y="Promedio",
                color="Sede",
                markers=True,
                text=traj["Promedio"].map(
                    lambda x: f"{x:.1f}"
                ),
                category_orders={
                    "Momento": ["Inicial", "Intermedia", "Final"]
                },
                custom_data=["N"],
                title=(
                    f"Trayectoria de 11° · "
                    f"{selected_metric.replace('Puntaje ', '')}"
                ),
            )
            fig_traj.update_traces(
                textposition="top center",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Momento: %{x}<br>"
                    "Promedio: %{y:.1f}<br>"
                    "n: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )
            fig_traj.update_layout(
                height=480,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Sede",
            )
            st.plotly_chart(
                fig_traj,
                use_container_width=True,
                key="evolution_grade11_trajectory",
            )

    if availability is not None and not availability.empty:
        with st.expander(
            "Ver disponibilidad de aplicaciones por sede y grado",
            expanded=False,
        ):
            av = availability.copy()
            av["Grado"] = av["Grado_num"].map(
                lambda g: f"{int(g)}°"
            )
            st.dataframe(
                av[["Grado", "Sede", "Momento", "Registros"]],
                use_container_width=True,
                hide_index=True,
            )


st.title("📊 Estado de llegada 2026")
st.caption(
    "Base precargada actualizada al 19 de octubre de 2026 · "
    "Análisis general, antigüedad, LATAM, actividad matemática y evolución 10°–11°."
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
evolution_summary, evolution_trajectory, evolution_availability = (
    read_evolution_10_11_data()
)

view_general, view_age, view_latam, view_math_activity, view_evolution = st.tabs(
    [
        "Análisis general",
        "Análisis por antigüedad ≤1 año",
        "LATAM vs Evaluar para Avanzar",
        "Actividad matemática vs EFA",
        "Evolución 10°–11°",
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
    )
