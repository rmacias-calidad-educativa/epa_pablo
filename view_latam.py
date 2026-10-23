from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
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
