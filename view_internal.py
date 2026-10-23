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
    grade_ranges = GRADE_APPLICABILITY

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
