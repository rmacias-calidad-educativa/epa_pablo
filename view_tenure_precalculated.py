"""Comparación precalculada, anónima y exclusiva de resultados EFA 2026."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import LEVEL_COLORS, LEVEL_ORDER, TESTS, MIN_PUBLIC_GROUP_SIZE

PATH = Path("data/efa_tenure_comparison_2026.csv")
GROUPS = ["≤1 año", "≥3 años"]
COLORS = {"≤1 año": "#1976B9", "≥3 años": "#E19A32"}
SITES = {"BAQ": "Portal de Genovés", "COT": "Cota", "MOS": "Mosquera",
         "TUN": "Tunja", "USAQ": "Usaquén", "ZIPA": "Zipaquirá"}
LEVEL_COLS = ["Nivel_1", "Nivel_2", "Nivel_3", "Nivel_4"]


@st.cache_data(show_spinner=False)
def read_efa_tenure_aggregate():
    if not PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(PATH)
    needed = {"Sede", "Grado_num", "Prueba", "Grupo", "Estudiantes",
              "Suma_puntaje", *LEVEL_COLS}
    if not needed.issubset(data.columns):
        return pd.DataFrame()
    for name in ["Grado_num", "Estudiantes", "Suma_puntaje", *LEVEL_COLS]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["Grado_num", "Estudiantes", "Suma_puntaje", *LEVEL_COLS])
    data = data[data["Grupo"].isin(GROUPS)
                & data["Estudiantes"].ge(MIN_PUBLIC_GROUP_SIZE)].copy()
    keys = ["Sede", "Grado_num", "Prueba"]
    eligible = data.groupby(keys)["Grupo"].nunique()
    data = data.set_index(keys).loc[
        lambda x: x.index.isin(eligible[eligible.eq(2)].index)
    ].reset_index()
    data = data[data[LEVEL_COLS].sum(axis=1).eq(data["Estudiantes"])].copy()
    eligible = data.groupby(keys)["Grupo"].nunique()
    data = data.set_index(keys).loc[
        lambda x: x.index.isin(eligible[eligible.eq(2)].index)
    ].reset_index()
    data["Puntaje"] = data["Suma_puntaje"] / data["Estudiantes"]
    data["Niveles_3_4"] = 100 * (data["Nivel_3"] + data["Nivel_4"]) / data["Estudiantes"]
    data["Grado"] = data["Grado_num"].astype(int).astype(str) + "°"
    data["Sede_nombre"] = data["Sede"].map(SITES).fillna(data["Sede"])
    return data


def render_efa_tenure_precalculated():
    st.markdown("## Evaluar para Avanzar · desempeño por permanencia")
    st.caption(
        "Contraste de EFA 2026 entre estudiantes con ≤1 año y ≥3 años. "
        "Los resultados individuales se cruzaron con el registro de "
        "antigüedad por documento y sede; solo se publican agregados anónimos."
    )
    base = read_efa_tenure_aggregate()
    if base.empty:
        st.warning("No está disponible el resumen precalculado de EFA por antigüedad.")
        return

    tests = [x for x in TESTS if x in set(base["Prueba"])]
    c1, c2, c3 = st.columns([1, 1.5, 1.5])
    with c1:
        test = st.selectbox("Prueba EFA", tests,
            index=tests.index("Matemáticas") if "Matemáticas" in tests else 0,
            key="precalc_efa_tenure_test")
    d = base[base["Prueba"].eq(test)].copy()
    sites = [x for x in SITES if x in set(d["Sede"])]
    grades = sorted(d["Grado_num"].dropna().astype(int).unique().tolist())
    with c2:
        sel_sites = st.multiselect("Sedes comparables", sites, default=sites,
            format_func=lambda x: SITES[x], key="precalc_efa_tenure_sites")
    with c3:
        sel_grades = st.multiselect("Grados comparables", grades, default=grades,
            format_func=lambda x: f"{x}°", key="precalc_efa_tenure_grades")
    d = d[d["Sede"].isin(sel_sites) & d["Grado_num"].isin(sel_grades)].copy()
    if d.empty:
        st.info("Selecciona alguna sede y grado con observaciones comparables.")
        return

    counts = d.groupby("Grupo")["Estudiantes"].sum()
    a, b, c = st.columns(3)
    a.metric("Observaciones EFA · ≤1 año", int(counts.get(GROUPS[0], 0)))
    b.metric("Observaciones EFA · ≥3 años", int(counts.get(GROUPS[1], 0)))
    c.metric("Cruces sede × grado comparables", len(d) // 2)
    st.caption(
        f"Se exige al menos {MIN_PUBLIC_GROUP_SIZE} estudiantes de cada grupo "
        "dentro de la misma sede, grado y prueba. Solo Mosquera tiene "
        "cruces suficientes en esta fuente; no se inventan comparaciones "
        "para sedes o grados sin tamaños de muestra adecuados."
    )

    for metric, title in [
        ("Puntaje", "Puntaje promedio EFA (% de aciertos)"),
        ("Niveles_3_4", "Porcentaje de estudiantes en niveles EFA 3 + 4"),
    ]:
        st.markdown("### " + title)
        fig = px.bar(
            d, x="Grado", y=metric, color="Grupo", barmode="group",
            facet_col="Sede_nombre", facet_col_wrap=2,
            color_discrete_map=COLORS,
            category_orders={
                "Grupo": GROUPS,
                "Grado": [f"{g}°" for g in sorted(sel_grades)],
                "Sede_nombre": [SITES[s] for s in sel_sites],
            },
            text=metric, custom_data=["Estudiantes"],
            labels={"Grado": "Grado", metric: title, "Grupo": "Antigüedad"},
        )
        fig.update_traces(
            texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False,
            hovertemplate=(
                "Grado %{x}<br>%{fullData.name}: %{y:.1f}%"
                "<br>n=%{customdata[0]:.0f}<extra></extra>"
            ),
        )
        fig.update_yaxes(range=[0, 110], ticksuffix="%")
        fig.for_each_annotation(
            lambda x: x.update(text=x.text.replace("Sede_nombre=", ""))
        )
        fig.update_layout(
            height=max(450, 350 * ((len(sel_sites) + 1) // 2)),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=20, r=20, t=90, b=40),
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"precalc_efa_tenure_{metric}")

    st.markdown("### Distribución en los cuatro niveles EFA")
    for site in sel_sites:
        site_d = d[d["Sede"].eq(site)].copy()
        if site_d.empty:
            continue
        st.markdown("#### " + SITES[site])
        site_d["Grupo_grado"] = site_d["Grado"] + " · " + site_d["Grupo"]
        order = [f"{g}° · {group}" for g in sorted(sel_grades) for group in GROUPS
                 if f"{g}° · {group}" in set(site_d["Grupo_grado"])]
        indexed = site_d.set_index("Grupo_grado").reindex(order)
        fig = go.Figure()
        for label, column in zip(LEVEL_ORDER, LEVEL_COLS):
            percent = (100 * indexed[column] / indexed["Estudiantes"]).fillna(0)
            fig.add_bar(
                x=percent, y=order, orientation="h", name=label,
                marker_color=LEVEL_COLORS[label],
                customdata=indexed[column].fillna(0),
                hovertemplate=(
                    "%{y}<br>" + label + ": %{x:.1f}%"
                    "<br>n=%{customdata:.0f}<extra></extra>"
                ),
            )
        fig.update_layout(
            barmode="stack", height=max(400, len(order) * 32 + 110),
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(order))),
            xaxis=dict(range=[0, 100], ticksuffix="%"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=12, r=20, t=65, b=30),
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"precalc_efa_tenure_levels_{site}")

    matrix = d.pivot_table(
        index=["Sede", "Grado_num"], columns="Grupo",
        values=["Estudiantes", "Puntaje", "Niveles_3_4"], aggfunc="first",
    )
    if not matrix.empty:
        st.markdown("### Brechas por sede y grado")
        st.dataframe(pd.DataFrame({
            "Sede": [SITES.get(x, x) for x in matrix.index.get_level_values("Sede")],
            "Grado": [f"{int(x)}°" for x in matrix.index.get_level_values("Grado_num")],
            "n ≤1": matrix[("Estudiantes", GROUPS[0])].astype(int).to_numpy(),
            "n ≥3": matrix[("Estudiantes", GROUPS[1])].astype(int).to_numpy(),
            "Puntaje ≤1 (%)": matrix[("Puntaje", GROUPS[0])].round(1).to_numpy(),
            "Puntaje ≥3 (%)": matrix[("Puntaje", GROUPS[1])].round(1).to_numpy(),
            "Brecha puntaje (pp)": (
                matrix[("Puntaje", GROUPS[1])] -
                matrix[("Puntaje", GROUPS[0])]
            ).round(1).to_numpy(),
            "Brecha niveles 3+4 (pp)": (
                matrix[("Niveles_3_4", GROUPS[1])] -
                matrix[("Niveles_3_4", GROUPS[0])]
            ).round(1).to_numpy(),
        }), hide_index=True, use_container_width=True)
    st.caption(
        "Brecha = ≥3 años menos ≤1 año. Son diferencias descriptivas "
        "entre cohortes, no estimaciones de efectos causales. "
        "Los grupos publicados cumplen el tamaño mínimo en cada grado."
    )
