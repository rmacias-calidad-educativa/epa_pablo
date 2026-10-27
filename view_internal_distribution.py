"""Distribución descriptiva de evaluación interna T1 y T2.

Los agregados disponibles están cruzados con EFA y NO contienen sede ni
campo de aprendizaje. No se atribuyen estos datos a sedes específicas.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

LEVELS = [
    "Progreso limitado", "Emergente", "En aceleración", "Avanzado",
]
COLORS = {
    "Progreso limitado": "#C93C3C",
    "Emergente": "#E9A529",
    "En aceleración": "#93C354",
    "Avanzado": "#2877BC",
}
TRIMESTERS = ["TRIMESTRE 1", "TRIMESTRE 2"]


def render_internal_distribution_only(distribution: pd.DataFrame):
    st.markdown("## Evaluación interna · distribución del desempeño")
    st.caption(
        "Comparación de T1 y T2 en barras apiladas al 100 %, por prueba EFA "
        "y grado. Cada distribución procede de estudiantes emparejados con EFA."
    )
    st.warning(
        "La base precargada de este módulo no tiene campos de aprendizaje "
        "ni sedes. Por eso esta vista utiliza únicamente las pruebas y los "
        "grados disponibles. No equivale a la distribución de todos los "
        "estudiantes de cada sede mostrada en la imagen de referencia."
    )

    if distribution is None or distribution.empty:
        st.info("No se encontraron datos de evaluación interna.")
        return

    base = distribution[
        distribution["Trimestre"].astype(str).isin(TRIMESTERS)
    ].copy()
    base["Estudiantes"] = pd.to_numeric(
        base["Estudiantes"], errors="coerce"
    ).fillna(0)
    base["Grado_num"] = pd.to_numeric(
        base["Grado_num"], errors="coerce"
    )
    base = base.dropna(subset=["Grado_num"])
    if base.empty:
        st.info("No existen resultados de T1 y T2 para esta selección.")
        return

    tests = sorted(base["Prueba"].dropna().astype(str).unique().tolist())
    grades = sorted(base["Grado_num"].dropna().astype(int).unique().tolist())
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_tests = st.multiselect(
            "Pruebas", tests, default=tests,
            key="internal_only_tests",
        )
    with c2:
        selected_grades = st.multiselect(
            "Grados", grades, default=grades,
            format_func=lambda value: f"{value}°",
            key="internal_only_grades",
        )
    if not selected_tests or not selected_grades:
        st.info("Selecciona al menos una prueba y un grado.")
        return

    base = base[
        base["Prueba"].isin(selected_tests)
        & base["Grado_num"].astype(int).isin(selected_grades)
    ].copy()

    # Sumar conteos, no porcentajes ya calculados: cada fila de EFA
    # representa un estrato distinto dentro de grado/prueba/trimestre.
    base = (
        base.groupby(
            ["Trimestre", "Prueba", "Grado_num", "Interna_nivel"],
            as_index=False,
        )["Estudiantes"].sum()
    )
    base["Panel"] = (
        base["Prueba"].astype(str) + " · "
        + base["Grado_num"].astype(int).astype(str) + "°"
    )
    totals = base.groupby(
        ["Trimestre", "Panel"]
    )["Estudiantes"].transform("sum")
    base["Porcentaje"] = (
        base["Estudiantes"].div(totals.where(totals.ne(0))).mul(100)
    )
    base["Etiqueta"] = base["Porcentaje"].map(
        lambda v: f"{v:.1f}%" if pd.notna(v) and v >= 5 else ""
    )

    for trimester in TRIMESTERS:
        st.markdown(
            "### " + ("Trimestre 1" if trimester.endswith("1")
                      else "Trimestre 2")
        )
        subset = base[base["Trimestre"] == trimester].copy()
        if subset.empty:
            st.info("Sin registros para este trimestre.")
            continue

        panels = sorted(subset["Panel"].unique().tolist())
        chart = px.bar(
            subset,
            x="Porcentaje", y="Panel",
            color="Interna_nivel",
            orientation="h",
            barmode="stack",
            text="Etiqueta",
            custom_data=["Estudiantes"],
            color_discrete_map=COLORS,
            category_orders={
                "Interna_nivel": LEVELS,
                "Panel": list(reversed(panels)),
            },
            labels={
                "Porcentaje": "Porcentaje de estudiantes",
                "Panel": "Prueba y grado",
                "Interna_nivel": "Desempeño",
            },
        )
        chart.update_traces(
            textposition="inside",
            hovertemplate=(
                "%{y}<br>%{fullData.name}<br>"
                "Porcentaje: %{x:.1f}%<br>"
                "Estudiantes: %{customdata[0]:.0f}<extra></extra>"
            ),
        )
        chart.update_xaxes(range=[0, 100], ticksuffix="%")
        chart.update_layout(
            height=max(440, 36 * len(panels) + 150),
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=10, r=18, t=85, b=30),
        )
        st.plotly_chart(
            chart, use_container_width=True,
            key=f"internal_only_distribution_{trimester}",
        )

    st.caption(
        "Denominador: estudiantes con registro emparejado EFA–evaluación "
        "interna en cada prueba, grado y trimestre. Un estudiante puede "
        "aparecer en más de una prueba; no se suman las barras como "
        "personas únicas."
    )
