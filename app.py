
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    ANTIGUEDAD_SITES,
    EVOLUTION_METRICS,
    GRADE_APPLICABILITY,
    IMPROVE_EFA_TESTS,
    LATAM_ORDER,
    LATAM_SCORE,
    LEVEL_COLORS,
    LEVEL_ORDER,
    MATH_ACTIVITY_METRICS,
    MATH_COMPETENCIES,
    MIN_PUBLIC_GROUP_SIZE,
    SITES,
    TESTS,
    YEAR,
)
from data_quality import suppress_small_groups
from loaders import (
    process_raw,
    read_default_data,
    read_evolution_10_11_data,
    read_excel,
    read_improve_efa_data,
    read_internal_efa_data,
    read_latam_efa_data,
    read_math_activity_data,
    read_student_story_3_9,
    read_student_story_profiles,
    read_student_story_support,
)
from views import (
    render_evolution_10_11,
    render_improve_efa_relation,
    render_internal_vs_efa,
    render_latam_efa_area,
    render_math_activity_vs_efa,
    render_student_story,
    render_test_page,
)

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


st.title("📊 Estado de llegada 2026")
st.caption(
    "Base precargada actualizada al 19 de octubre de 2026 · "
    "Análisis general, antigüedad, LATAM, actividad matemática, evolución, "
    "relación EFA–Improve, evaluación interna e historia longitudinal. "
    f"Control de divulgación: las vistas sensibles suprimen grupos con n < "
    f"{MIN_PUBLIC_GROUP_SIZE}."
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
    student_story_internal,
    student_story_latam_10_11,
) = read_student_story_support()

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
        student_story_internal,
        student_story_latam_10_11,
    )
