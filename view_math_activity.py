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


from view_utils import style_clean_axes


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
