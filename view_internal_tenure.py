"""Contraste precalculado y anónimo de evaluación interna T1/T2 por antigüedad."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PATH = Path("data/internal_tenure_contrast_2026.csv")
GROUPS = ["≤1 año", "≥3 años"]
COLORS = {"≤1 año": "#2683C6", "≥3 años": "#E49A37"}
SITES = {"BAQ": "Portal de Genovés", "COT": "Cota",
         "MOS": "Mosquera", "TUN": "Tunja", "USAQ": "Usaquén",
         "ZIPA": "Zipaquirá"}
GRADES = ["Todos", "Prejardín", "Jardín", "Transición",
          *[f"{i}°" for i in range(1, 12)]]


@st.cache_data(show_spinner=False)
def load_internal_tenure() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame()
    d = pd.read_csv(PATH)
    required = {"Sede", "Trimestre", "Grado", "Area", "Grupo", "n",
                "Alto", "Avanzado"}
    if not required.issubset(d.columns):
        return pd.DataFrame()
    for c in ["n", "Alto", "Avanzado"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["n", "Alto", "Avanzado"])
    d = d[d["n"].ge(5) & d["Grupo"].isin(GROUPS)].copy()
    keys = ["Sede", "Trimestre", "Grado", "Area"]
    valid = d.groupby(keys)["Grupo"].nunique()
    return d.set_index(keys).loc[
        lambda x: x.index.isin(valid[valid == 2].index)
    ].reset_index()


def render_internal_tenure() -> None:
    st.markdown("## Evaluación interna · ≤1 año frente a ≥3 años")
    st.caption(
        "Resultados T1 y T2, Escala DESPUÉS DE POTENCIACIÓN. "
        "Datos emparejados por documento y sede, publicados solo como "
        "agregados anónimos. No necesitas cargar archivos."
    )
    d = load_internal_tenure()
    if d.empty:
        st.warning("No se encontraron estadísticas precargadas de evaluación interna.")
        return

    grades = [g for g in GRADES if g in set(d["Grado"])]
    c1, c2, c3 = st.columns([1, 1.8, 2])
    with c1:
        grade = st.selectbox(
            "Grado", grades,
            format_func=lambda x: "Todos los grados" if x == "Todos" else x,
            key="tenure_internal_grade",
        )
    d = d[d["Grado"] == grade].copy()
    options = [s for s in SITES if s in set(d["Sede"])]
    with c2:
        sites = st.multiselect(
            "Sedes comparables", options, default=options,
            format_func=lambda s: SITES[s], key="tenure_internal_sites",
        )
    with c3:
        metric = st.selectbox(
            "Desempeño", ["En aceleración + Avanzado", "Solo Avanzado"],
            key="tenure_internal_metric",
        )

    d = d[d["Sede"].isin(sites)].copy()
    available = sorted(d["Area"].unique().tolist())
    areas = st.multiselect(
        "Áreas", available, default=available, key="tenure_internal_areas",
    )
    d = d[d["Area"].isin(areas)].copy()
    if d.empty:
        st.info("No hay cohortes comparables con esos filtros.")
        return

    column = "Alto" if metric.startswith("En aceleración") else "Avanzado"
    d["Pct"] = d[column].div(d["n"]).mul(100)
    st.caption(
        "Las dos barras de cada Área representan los porcentajes de "
        "estudiantes de cada grupo. Solo se incluyen comparaciones con "
        "al menos cinco estudiantes en ambas cohortes."
    )
    if grade == "Todos":
        st.info(
            "La vista de todos los grados no ajusta por diferencias de "
            "composición entre las cohortes. Selecciona un grado específico "
            "para una comparación más homogénea."
        )

    for trimester, short in [("TRIMESTRE 1", "T1"), ("TRIMESTRE 2", "T2")]:
        st.markdown(f"### {short} · {metric}")
        sub = d[d["Trimestre"] == trimester]
        visible = [s for s in sites if s in set(sub["Sede"])]
        if not visible:
            st.info(f"Sin comparaciones publicables en {short}.")
            continue
        cols = min(2, len(visible))
        rows = (len(visible) + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[SITES[s] for s in visible],
            horizontal_spacing=0.25 if cols == 2 else 0.05,
            vertical_spacing=0.13 if rows > 1 else 0.04,
        )
        max_bars = 0
        for idx, site in enumerate(visible):
            row, col = divmod(idx, cols)
            part = sub[sub["Sede"] == site].copy()
            order = sorted(part["Area"].unique().tolist())
            max_bars = max(max_bars, len(order))
            for group in GROUPS:
                group_data = part[part["Grupo"] == group].set_index("Area").reindex(order)
                pct = group_data["Pct"].fillna(0).tolist()
                counts = group_data["n"].fillna(0).tolist()
                fig.add_trace(
                    go.Bar(
                        x=pct, y=order, orientation="h", name=group,
                        marker_color=COLORS[group],
                        text=[f"{v:.0f}%" if v >= 8 else "" for v in pct],
                        textposition="auto", customdata=counts,
                        hovertemplate=(
                            "%{y}<br>" + group + ": %{x:.1f}%"
                            "<br>Estudiantes: %{customdata:.0f}<extra></extra>"
                        ),
                        showlegend=idx == 0,
                    ),
                    row=row + 1, col=col + 1,
                )
            fig.update_xaxes(range=[0, 105], ticksuffix="%",
                             row=row + 1, col=col + 1)
            fig.update_yaxes(
                categoryorder="array", categoryarray=list(reversed(order)),
                tickfont=dict(size=10), row=row + 1, col=col + 1,
            )
        fig.update_layout(
            barmode="group",
            height=rows * max(460, max_bars * 42 + 135),
            legend=dict(orientation="h", y=1.06, x=0),
            margin=dict(l=12, r=18, t=90, b=40),
        )
        st.plotly_chart(
            fig, use_container_width=True,
            key=f"tenure_internal_plot_{short}",
        )

    st.markdown("### Tabla de diferencias observadas")
    pivot = d.pivot_table(
        index=["Sede", "Trimestre", "Grado", "Area"],
        columns="Grupo", values=["n", "Pct"], aggfunc="first",
    )
    if not pivot.empty:
        result = pd.DataFrame({
            "Sede": [SITES.get(s, s) for s in pivot.index.get_level_values("Sede")],
            "Trimestre": pivot.index.get_level_values("Trimestre").str.replace(
                "TRIMESTRE ", "T", regex=False,
            ),
            "Grado": pivot.index.get_level_values("Grado"),
            "Área": pivot.index.get_level_values("Area"),
            "n ≤1": pivot[("n", "≤1 año")].astype(int).values,
            "n ≥3": pivot[("n", "≥3 años")].astype(int).values,
            "% ≤1": pivot[("Pct", "≤1 año")].round(1).values,
            "% ≥3": pivot[("Pct", "≥3 años")].round(1).values,
            "Brecha (pp)": (
                pivot[("Pct", "≥3 años")] - pivot[("Pct", "≤1 año")]
            ).round(1).values,
        })
        st.dataframe(
            result.sort_values(["Sede", "Trimestre", "Área"]),
            use_container_width=True, hide_index=True,
        )

    missing = [SITES[s] for s in ["COT", "TUN"] if s not in set(load_internal_tenure()["Sede"])]
    if missing:
        st.caption(
            "Sin comparaciones publicables para " + " y ".join(missing) +
            ": no hay al menos cinco estudiantes en ambos grupos."
        )
    st.caption(
        "Brecha = porcentaje de ≥3 años menos porcentaje de ≤1 año, "
        "en puntos porcentuales. Una diferencia no demuestra un efecto "
        "causal de la permanencia; también pueden intervenir factores "
        "de selección, grado y contexto."
    )
