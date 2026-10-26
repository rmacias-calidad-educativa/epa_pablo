
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import MIN_PUBLIC_GROUP_SIZE, TESTS
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

LOAD_ERRORS: list[str] = []


def safe_secondary_load(
    label: str,
    loader,
    outputs: int,
):
    try:
        return loader()
    except Exception as exc:
        LOAD_ERRORS.append(f"{label}: {exc}")
        if outputs == 1:
            return pd.DataFrame()
        return tuple(pd.DataFrame() for _ in range(outputs))


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
    try:
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
    except Exception as exc:
        st.error(
            "No fue posible cargar la base principal de EFA. "
            f"Detalle: {exc}"
        )
        st.stop()

    if attempts.empty:
        st.info("Carga el Excel para visualizar los resultados.")
        st.stop()

latam_efa_data, latam_efa_summary = safe_secondary_load(
    "LATAM vs EFA",
    read_latam_efa_data,
    2,
)
(
    math_activity_stats,
    math_activity_bins,
    math_activity_summary,
) = safe_secondary_load(
    "Actividad matemática vs EFA",
    read_math_activity_data,
    3,
)
(
    evolution_summary,
    evolution_trajectory,
    evolution_availability,
    evolution_observed,
) = safe_secondary_load(
    "Evolución 10°–11°",
    read_evolution_10_11_data,
    4,
)
(
    improve_efa_levels,
    improve_efa_quality,
    improve_efa_growth,
) = safe_secondary_load(
    "EFA vs Improve",
    read_improve_efa_data,
    3,
)
(
    internal_efa_summary,
    internal_efa_distribution,
    internal_efa_coverage,
    internal_efa_agreement,
    internal_efa_reverse,
) = safe_secondary_load(
    "Evaluación interna vs EFA",
    read_internal_efa_data,
    5,
)
student_story_profiles = safe_secondary_load(
    "Historia longitudinal 10°–11°",
    read_student_story_profiles,
    1,
)
student_story_3_9 = safe_secondary_load(
    "Historia longitudinal 3°–9°",
    read_student_story_3_9,
    1,
)
(
    student_story_internal,
    student_story_latam_10_11,
) = safe_secondary_load(
    "Soportes de historia longitudinal",
    read_student_story_support,
    2,
)

if LOAD_ERRORS:
    st.warning(
        "Algunas fuentes secundarias no pudieron cargarse. "
        "Las demás vistas siguen disponibles."
    )
    with st.expander("Ver detalle de fuentes no disponibles"):
        for error in LOAD_ERRORS:
            st.write(f"• {error}")

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
        "Esta ventana incluye a todos los estudiantes de Portal de Génova (BAQ) y Usaquén (USAQ), "
        "y a los estudiantes de COT, MOS, TUN y ZIPA con 1 año o menos. Las gráficas y "
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
                        "BAQ y USAQ: todos · COT, MOS, TUN y ZIPA: ≤1 año"
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
        internal_efa_summary,
    )
