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


def _relation_strength(value: float) -> str:
    if not np.isfinite(value):
        return "Sin dato"
    v = abs(float(value))
    if v < 0.20:
        return "Muy débil"
    if v < 0.40:
        return "Débil"
    if v < 0.60:
        return "Moderada"
    if v < 0.80:
        return "Fuerte"
    return "Muy fuerte"


def render_improve_efa_relation(
    levels: pd.DataFrame,
    quality: pd.DataFrame,
    growth: pd.DataFrame,
):
    st.markdown("## Evaluar para Avanzar vs Improve")
    st.caption(
        "Promedio de la primera aplicación de Improve según el nivel de "
        "desempeño alcanzado por los mismos estudiantes en EFA."
    )

    if levels.empty:
        st.warning(
            "No está disponible el resumen por niveles EFA."
        )
        return

    st.info(
        "El emparejamiento es estricto: documento exacto + mismo grado + "
        "misma sede, excluyendo registros marcados para revisión. "
        "Los niveles EFA son: Nivel 1 = 0–25%, Nivel 2 = >25–50%, "
        "Nivel 3 = >50–75% y Nivel 4 = >75–100%. "
        "Improve corresponde exclusivamente a la primera aplicación."
    )

    level_order = ["Nivel 1", "Nivel 2", "Nivel 3", "Nivel 4"]
    test_order = IMPROVE_EFA_TESTS
    site_order = [
        "Cota",
        "Mosquera",
        "Tunja",
        "Zipaquirá",
        "BAQ",
        "Usaquén",
    ]

    for grade in [10, 11]:
        grade_levels = levels[
            (levels["grado_num"] == grade)
            & levels["Prueba"].astype(str).isin(test_order)
        ].copy()

        if grade_levels.empty:
            continue

        st.divider()
        st.markdown(f"## {grade}°")

        q_grade = (
            quality[
                (quality["grado_num"] == grade)
                & (quality["Sede"].astype(str) == "RED")
            ]
            if quality is not None and not quality.empty
            else pd.DataFrame()
        )

        if not q_grade.empty:
            qrow = q_grade.iloc[0]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric(
                "Estudiantes Improve EV1",
                f"{int(qrow['Improve_EV1_estudiantes'])}",
            )
            q2.metric(
                "Emparejados estrictos",
                f"{int(qrow['Emparejados_identidad'])}",
            )
            q3.metric(
                "Cobertura del cruce",
                f"{float(qrow['Cobertura_identidad_pct']):.1f}%",
            )
            q4.metric(
                "No emparejados",
                f"{int(qrow['No_emparejados'])}",
            )

        st.markdown("### Promedio Improve por nivel de EFA · RED")
        st.caption(
            "La lectura central es vertical: dentro de cada prueba, observa "
            "cómo cambia el promedio de Improve al pasar de Nivel 1 a Nivel 4 "
            "en Evaluar para Avanzar."
        )

        red = grade_levels[
            grade_levels["Sede"].astype(str) == "RED"
        ].copy()
        red["Nivel_EFA"] = pd.Categorical(
            red["Nivel_EFA"],
            categories=level_order,
            ordered=True,
        )
        red["Prueba"] = pd.Categorical(
            red["Prueba"],
            categories=test_order,
            ordered=True,
        )
        red = red.sort_values(["Prueba", "Orden"])

        if not red.empty:
            fig_red = px.line(
                red,
                x="Nivel_EFA",
                y="Promedio_Improve",
                facet_col="Prueba",
                facet_col_wrap=3,
                markers=True,
                text=red["Promedio_Improve"].map(
                    lambda x: f"{x:.1f}"
                ),
                custom_data=["n", "Promedio_EFA"],
                category_orders={
                    "Nivel_EFA": level_order,
                    "Prueba": test_order,
                },
                title=(
                    f"Primera aplicación Improve según nivel EFA · RED, "
                    f"{grade}°"
                ),
            )
            fig_red.update_yaxes(
                range=[0, 100],
                title="Promedio Improve",
            )
            fig_red.update_xaxes(
                title=None,
                categoryorder="array",
                categoryarray=level_order,
            )
            fig_red.update_traces(
                textposition="top center",
                cliponaxis=False,
                hovertemplate=(
                    "Nivel EFA: %{x}<br>"
                    "Promedio Improve: %{y:.1f}<br>"
                    "n: %{customdata[0]}<br>"
                    "Promedio EFA del grupo: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
            fig_red.for_each_annotation(
                lambda a: a.update(
                    text=a.text.replace("Prueba=", "")
                )
            )
            fig_red.update_layout(
                height=820,
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(
                fig_red,
                use_container_width=True,
                key=f"improve_efa_levels_red_{grade}",
            )

            matrix = red.pivot_table(
                index="Prueba",
                columns="Nivel_EFA",
                values="Promedio_Improve",
                aggfunc="first",
                observed=False,
            ).reindex(index=test_order, columns=level_order)

            n_matrix = red.pivot_table(
                index="Prueba",
                columns="Nivel_EFA",
                values="n",
                aggfunc="first",
                observed=False,
            ).reindex(index=test_order, columns=level_order)

            display_rows = []
            for test in test_order:
                row = {"Prueba": test}
                for level in level_order:
                    val = matrix.loc[test, level]
                    n_val = n_matrix.loc[test, level]
                    if pd.isna(val):
                        row[level] = "Sin casos"
                    else:
                        row[level] = (
                            f"{val:.1f} (n={int(n_val)})"
                            if not pd.isna(n_val)
                            else f"{val:.1f}"
                        )
                display_rows.append(row)

            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Comportamiento por sede")
        st.caption(
            "Cada línea corresponde a una sede. Esto permite verificar si la "
            "progresión entre niveles EFA e Improve se mantiene también dentro "
            "de los colegios y no solo en el promedio de la RED."
        )

        sites = grade_levels[
            grade_levels["Sede"].astype(str) != "RED"
        ].copy()
        sites = sites[
            sites["Sede"].astype(str).isin(site_order)
        ]
        sites["Nivel_EFA"] = pd.Categorical(
            sites["Nivel_EFA"],
            categories=level_order,
            ordered=True,
        )
        sites["Prueba"] = pd.Categorical(
            sites["Prueba"],
            categories=test_order,
            ordered=True,
        )
        sites["Sede"] = pd.Categorical(
            sites["Sede"],
            categories=site_order,
            ordered=True,
        )
        sites = sites.sort_values(
            ["Prueba", "Sede", "Orden"]
        )

        if not sites.empty:
            fig_sites = px.line(
                sites,
                x="Nivel_EFA",
                y="Promedio_Improve",
                color="Sede",
                facet_col="Prueba",
                facet_col_wrap=3,
                markers=True,
                custom_data=["n", "Promedio_EFA"],
                category_orders={
                    "Nivel_EFA": level_order,
                    "Prueba": test_order,
                    "Sede": site_order,
                },
                title=(
                    f"Promedio Improve por nivel EFA y sede · {grade}°"
                ),
            )
            fig_sites.update_yaxes(
                range=[0, 100],
                title="Promedio Improve",
            )
            fig_sites.update_xaxes(
                title=None,
                categoryorder="array",
                categoryarray=level_order,
            )
            fig_sites.update_traces(
                hovertemplate=(
                    "Sede: %{fullData.name}<br>"
                    "Nivel EFA: %{x}<br>"
                    "Promedio Improve: %{y:.1f}<br>"
                    "n: %{customdata[0]}<br>"
                    "Promedio EFA del grupo: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
            fig_sites.for_each_annotation(
                lambda a: a.update(
                    text=a.text.replace("Prueba=", "")
                )
            )
            fig_sites.update_layout(
                height=840,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Sede",
            )
            st.plotly_chart(
                fig_sites,
                use_container_width=True,
                key=f"improve_efa_levels_sites_{grade}",
            )

        if growth is not None and not growth.empty:
            st.markdown(
                "### Crecimiento en Improve según nivel de Evaluar para Avanzar"
            )
            final_label = (
                "Evaluación 2"
                if grade == 10
                else "Evaluación 3"
            )
            st.caption(
                f"Se comparan los mismos estudiantes entre Evaluación 1 y "
                f"{final_label} de Improve. Los grupos se forman según el "
                "nivel obtenido en EFA para cada prueba."
            )

            growth_grade = growth[
                (growth["grado_num"] == grade)
                & (growth["Sede"].astype(str) == "RED")
                & growth["Prueba"].astype(str).isin(test_order)
            ].copy()

            if not growth_grade.empty:
                growth_grade["Nivel_EFA"] = pd.Categorical(
                    growth_grade["Nivel_EFA"],
                    categories=level_order,
                    ordered=True,
                )
                growth_grade["Prueba"] = pd.Categorical(
                    growth_grade["Prueba"],
                    categories=test_order,
                    ordered=True,
                )
                growth_grade = growth_grade.sort_values(
                    ["Prueba", "Orden"]
                )

                fig_growth = px.bar(
                    growth_grade,
                    x="Prueba",
                    y="Cambio_Improve",
                    color="Nivel_EFA",
                    barmode="group",
                    text=growth_grade["Cambio_Improve"].map(
                        lambda x: f"{x:+.1f}"
                    ),
                    custom_data=["n", "Improve_inicial", "Improve_final"],
                    category_orders={
                        "Prueba": test_order,
                        "Nivel_EFA": level_order,
                    },
                    title=(
                        f"Cambio promedio en Improve según nivel EFA · "
                        f"{grade}°"
                    ),
                )
                fig_growth.add_hline(y=0, line_dash="dot")
                fig_growth.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "Prueba: %{x}<br>"
                        "Nivel EFA: %{fullData.name}<br>"
                        "Cambio: %{y:+.1f}<br>"
                        "n: %{customdata[0]}<br>"
                        "Inicial: %{customdata[1]:.1f}<br>"
                        "Final: %{customdata[2]:.1f}"
                        "<extra></extra>"
                    ),
                )
                fig_growth.update_layout(
                    height=540,
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Nivel EFA",
                )
                st.plotly_chart(
                    fig_growth,
                    use_container_width=True,
                    key=f"improve_growth_by_efa_level_{grade}",
                )

                st.markdown("#### De cuánto a cuánto cambió cada nivel")

                long_growth = pd.concat(
                    [
                        growth_grade[
                            [
                                "Prueba",
                                "Nivel_EFA",
                                "n",
                                "Improve_inicial",
                            ]
                        ]
                        .rename(
                            columns={
                                "Improve_inicial": "Promedio Improve"
                            }
                        )
                        .assign(Momento="Evaluación 1"),
                        growth_grade[
                            [
                                "Prueba",
                                "Nivel_EFA",
                                "n",
                                "Improve_final",
                            ]
                        ]
                        .rename(
                            columns={
                                "Improve_final": "Promedio Improve"
                            }
                        )
                        .assign(Momento=final_label),
                    ],
                    ignore_index=True,
                )

                fig_initial_final = px.line(
                    long_growth,
                    x="Momento",
                    y="Promedio Improve",
                    color="Nivel_EFA",
                    facet_col="Prueba",
                    facet_col_wrap=3,
                    markers=True,
                    text=long_growth["Promedio Improve"].map(
                        lambda x: f"{x:.1f}"
                    ),
                    custom_data=["n"],
                    category_orders={
                        "Prueba": test_order,
                        "Nivel_EFA": level_order,
                        "Momento": ["Evaluación 1", final_label],
                    },
                    title=(
                        f"Puntaje Improve inicial vs final por nivel EFA · "
                        f"{grade}°"
                    ),
                )
                fig_initial_final.update_yaxes(
                    range=[0, 100],
                    title="Promedio Improve",
                )
                fig_initial_final.update_xaxes(title=None)
                fig_initial_final.update_traces(
                    textposition="top center",
                    cliponaxis=False,
                    hovertemplate=(
                        "Nivel EFA: %{fullData.name}<br>"
                        "Momento: %{x}<br>"
                        "Promedio Improve: %{y:.1f}<br>"
                        "n: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                )
                fig_initial_final.for_each_annotation(
                    lambda a: a.update(
                        text=a.text.replace("Prueba=", "")
                    )
                )
                fig_initial_final.update_layout(
                    height=840,
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Nivel EFA",
                )
                st.plotly_chart(
                    fig_initial_final,
                    use_container_width=True,
                    key=f"improve_initial_final_by_efa_level_{grade}",
                )

                best_rows = []
                for test in test_order:
                    dt = growth_grade[
                        growth_grade["Prueba"].astype(str) == test
                    ].copy()
                    if dt.empty:
                        continue
                    best = dt.sort_values(
                        "Cambio_Improve",
                        ascending=False,
                    ).iloc[0]
                    best_rows.append({
                        "Prueba": test,
                        "Nivel con mejor cambio": str(
                            best["Nivel_EFA"]
                        ),
                        "Cambio promedio": best["Cambio_Improve"],
                        "Inicial": best["Improve_inicial"],
                        "Final": best["Improve_final"],
                        "n": int(best["n"]),
                    })

                if best_rows:
                    best_table = pd.DataFrame(best_rows)
                    best_display = best_table.copy()
                    best_display["Cambio promedio"] = best_display[
                        "Cambio promedio"
                    ].map(lambda x: f"{x:+.1f}")
                    best_display["Inicial"] = best_display[
                        "Inicial"
                    ].map(lambda x: f"{x:.1f}")
                    best_display["Final"] = best_display[
                        "Final"
                    ].map(lambda x: f"{x:.1f}")
                    st.markdown(
                        "#### Nivel EFA con mejor cambio por prueba"
                    )
                    st.dataframe(
                        best_display,
                        use_container_width=True,
                        hide_index=True,
                    )

                growth_display = growth_grade[
                    [
                        "Prueba",
                        "Nivel_EFA",
                        "n",
                        "Improve_inicial",
                        "Improve_final",
                        "Cambio_Improve",
                        "Pct_mejora",
                    ]
                ].copy()
                growth_display = growth_display.rename(
                    columns={
                        "Nivel_EFA": "Nivel EFA",
                        "n": "Estudiantes",
                        "Improve_inicial": "Inicial",
                        "Improve_final": "Final",
                        "Cambio_Improve": "Cambio",
                        "Pct_mejora": "% que mejora",
                    }
                )
                for col in ["Inicial", "Final"]:
                    growth_display[col] = growth_display[col].map(
                        lambda x: f"{x:.1f}"
                    )
                growth_display["Cambio"] = growth_display[
                    "Cambio"
                ].map(lambda x: f"{x:+.1f}")
                growth_display["% que mejora"] = growth_display[
                    "% que mejora"
                ].map(lambda x: f"{x:.1f}%")

                st.dataframe(
                    growth_display,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Los cambios con grupos muy pequeños deben interpretarse "
                    "con cautela. Se muestra el tamaño de cada grupo (n) junto "
                    "con el resultado."
                )

        st.markdown("### Tabla completa · prueba × nivel × sede")
        table = grade_levels[
            grade_levels["Prueba"].astype(str).isin(test_order)
        ].copy()
        table["Nivel_EFA"] = pd.Categorical(
            table["Nivel_EFA"],
            categories=level_order,
            ordered=True,
        )
        table["Prueba"] = pd.Categorical(
            table["Prueba"],
            categories=test_order,
            ordered=True,
        )
        table = table.sort_values(
            ["Sede", "Prueba", "Orden"]
        )

        table_display = table[
            [
                "Sede",
                "Prueba",
                "Nivel_EFA",
                "n",
                "Promedio_EFA",
                "Promedio_Improve",
            ]
        ].copy()
        table_display = table_display.rename(
            columns={
                "Nivel_EFA": "Nivel EFA",
                "n": "Estudiantes",
                "Promedio_EFA": "Promedio EFA",
                "Promedio_Improve": "Promedio Improve",
            }
        )
        table_display["Promedio EFA"] = table_display[
            "Promedio EFA"
        ].map(lambda x: f"{x:.1f}%")
        table_display["Promedio Improve"] = table_display[
            "Promedio Improve"
        ].map(lambda x: f"{x:.1f}/100")

        st.dataframe(
            table_display,
            use_container_width=True,
            hide_index=True,
        )
