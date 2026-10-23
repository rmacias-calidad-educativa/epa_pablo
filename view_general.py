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


from view_utils import grade_label, style_clean_axes


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
    trend = trend[
        trend["Estudiantes"] >= MIN_PUBLIC_GROUP_SIZE
    ].copy()
    if trend.empty:
        return

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
        ranking = suppress_small_groups(
            ranking,
            "Estudiantes",
            MIN_PUBLIC_GROUP_SIZE,
        )
        if ranking.empty:
            st.info(
                "No hay grupos con al menos "
                f"{MIN_PUBLIC_GROUP_SIZE} estudiantes para mostrar."
            )
            return

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
        ranking = suppress_small_groups(
            ranking,
            "Estudiantes",
            MIN_PUBLIC_GROUP_SIZE,
        )

    st.caption(
        f"Por privacidad, no se muestran combinaciones con menos de "
        f"{MIN_PUBLIC_GROUP_SIZE} estudiantes."
    )
    if ranking.empty:
        st.info(
            "No hay grupos con tamaño suficiente para mostrar "
            "colegios de origen."
        )
        return

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
