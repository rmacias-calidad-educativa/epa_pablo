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


def _story_level_from_pct(value) -> str:
    if pd.isna(value):
        return "Sin dato"
    value = float(value)
    if value <= 25:
        return "Nivel 1"
    if value <= 50:
        return "Nivel 2"
    if value <= 75:
        return "Nivel 3"
    return "Nivel 4"


def _render_internal_story(
    story_internal: pd.DataFrame,
    grade: int,
    test: str,
    level: str,
    key: str,
):
    if story_internal is None or story_internal.empty:
        return

    d = story_internal[
        (story_internal["grado_num"] == grade)
        & (
            story_internal["Prueba_filtro"].astype(str)
            == test
        )
        & (
            story_internal["Nivel_EFA_filtro"].astype(str)
            == level
        )
    ].copy()

    if d.empty:
        return

    trim_order = ["TRIMESTRE 1", "TRIMESTRE 2"]
    d = d[
        d["Trimestre"].astype(str).isin(trim_order)
    ].copy()
    d["Trimestre"] = pd.Categorical(
        d["Trimestre"],
        categories=trim_order,
        ordered=True,
    )
    d = d.sort_values("Trimestre")

    st.markdown(
        "### Evaluación interna · misma cohorte de máxima cobertura"
    )
    fig = px.line(
        d,
        x="Trimestre",
        y="Promedio_interno",
        markers=True,
        text=d["Promedio_interno"].map(
            lambda x: f"{x:.2f}"
        ),
        custom_data=["Estudiantes"],
        category_orders={"Trimestre": trim_order},
        title=(
            f"Desempeño interno · {level} de {test} · T1 → T2"
        ),
    )
    fig.update_yaxes(
        range=[1, 4.15],
        title="Desempeño interno",
        tickmode="array",
        tickvals=[1, 2, 3, 4],
        ticktext=[
            "Progreso limitado",
            "Emergente",
            "En aceleración",
            "Avanzado",
        ],
    )
    fig.update_xaxes(title=None)
    fig.update_traces(
        textposition="top center",
        cliponaxis=False,
        hovertemplate=(
            "Trimestre: %{x}<br>"
            "Desempeño interno: %{y:.2f}<br>"
            "n: %{customdata[0]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )



def _filter_story_scope(
    df: pd.DataFrame,
    grade,
    level: str,
    test: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if grade != "Todos":
        out = out[out["grado_num"] == int(grade)]
    if level != "Todos":
        out = out[
            out["Nivel_EFA_filtro"].astype(str) == str(level)
        ]
    if test != "Todas":
        out = out[
            out["Prueba_filtro"].astype(str) == str(test)
        ]
    return out


def _story_profile_catalog(
    profiles: pd.DataFrame,
    profiles_3_9: pd.DataFrame,
) -> pd.DataFrame:
    parts = []

    if profiles_3_9 is not None and not profiles_3_9.empty:
        low = profiles_3_9[
            (profiles_3_9["Fuente"].astype(str) == "Evaluar para Avanzar")
            & (
                profiles_3_9["Resultado"].astype(str)
                == profiles_3_9["Prueba_filtro"].astype(str)
            )
        ].copy()
        if not low.empty:
            low = low[
                [
                    "grado_num",
                    "Prueba_filtro",
                    "Nivel_EFA_filtro",
                    "Grupo_EFA_n",
                    "Fuentes_maximas",
                    "Fuentes_requeridas",
                    "Valor",
                ]
            ].rename(
                columns={
                    "Grupo_EFA_n": "Grupo_n",
                    "Valor": "EFA_ancla",
                }
            )
            low["Tramo"] = "3°–9°"
            parts.append(low)

    if profiles is not None and not profiles.empty:
        high = profiles[
            [
                "grado_num",
                "Prueba_filtro",
                "Nivel_EFA_filtro",
                "Candidatos_completos",
                "Fuentes_maximas",
                "Fuentes_requeridas",
                "EFA_seleccionada_mediana",
            ]
        ].copy().rename(
            columns={
                "Candidatos_completos": "Grupo_n",
                "EFA_seleccionada_mediana": "EFA_ancla",
            }
        )
        high["Tramo"] = "10°–11°"
        parts.append(high)

    if not parts:
        return pd.DataFrame()

    catalog = pd.concat(parts, ignore_index=True)
    catalog["Grupo_n"] = pd.to_numeric(
        catalog["Grupo_n"], errors="coerce"
    ).fillna(0).astype(int)
    catalog["EFA_ancla"] = pd.to_numeric(
        catalog["EFA_ancla"], errors="coerce"
    )
    catalog["Fuentes_maximas"] = pd.to_numeric(
        catalog["Fuentes_maximas"], errors="coerce"
    ).fillna(0).astype(int)
    return catalog.drop_duplicates(
        subset=[
            "grado_num",
            "Prueba_filtro",
            "Nivel_EFA_filtro",
        ]
    )


def _render_story_overview(
    profiles: pd.DataFrame,
    profiles_3_9: pd.DataFrame,
    story_internal: pd.DataFrame,
    story_latam_10_11: pd.DataFrame,
    selected_grade,
    selected_level: str,
    selected_test: str,
):
    catalog = _story_profile_catalog(
        profiles,
        profiles_3_9,
    )
    catalog = _filter_story_scope(
        catalog,
        selected_grade,
        selected_level,
        selected_test,
    )

    if catalog.empty:
        st.warning(
            "No hay cohortes de máxima cobertura para esta selección."
        )
        return

    st.markdown("### Panorama de máxima evidencia")
    st.caption(
        "Cada punto representa una cohorte agregada definida por "
        "grado × nivel EFA × prueba EFA. No se promedian cohortes "
        "superpuestas entre sí; se muestran todas las evidencias "
        "disponibles para conservar su interpretación."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Cohortes disponibles",
        f"{len(catalog):,}".replace(",", "."),
    )
    k2.metric(
        "Grados representados",
        f"{catalog['grado_num'].nunique()}",
    )
    k3.metric(
        "Pruebas EFA",
        f"{catalog['Prueba_filtro'].nunique()}",
    )
    k4.metric(
        "Máximo de fuentes",
        f"{int(catalog['Fuentes_maximas'].max())}",
    )

    # Mapa de disponibilidad de fuentes por cohorte.
    evidence_parts = []

    low_scope = _filter_story_scope(
        profiles_3_9,
        selected_grade,
        selected_level,
        selected_test,
    )
    if not low_scope.empty:
        low_sources = low_scope[
            [
                "grado_num",
                "Prueba_filtro",
                "Nivel_EFA_filtro",
                "Fuente",
            ]
        ].drop_duplicates()
        evidence_parts.append(low_sources)

    high_scope = _filter_story_scope(
        profiles,
        selected_grade,
        selected_level,
        selected_test,
    )
    if not high_scope.empty:
        base_keys = high_scope[
            [
                "grado_num",
                "Prueba_filtro",
                "Nivel_EFA_filtro",
            ]
        ].drop_duplicates()
        for source in ["Evaluar para Avanzar", "Improve"]:
            tmp = base_keys.copy()
            tmp["Fuente"] = source
            evidence_parts.append(tmp)

    internal_scope = _filter_story_scope(
        story_internal,
        selected_grade,
        selected_level,
        selected_test,
    )
    if not internal_scope.empty:
        tmp = internal_scope[
            [
                "grado_num",
                "Prueba_filtro",
                "Nivel_EFA_filtro",
            ]
        ].drop_duplicates()
        tmp["Fuente"] = "Evaluación interna"
        evidence_parts.append(tmp)

    latam_high_scope = _filter_story_scope(
        story_latam_10_11,
        selected_grade,
        selected_level,
        selected_test,
    )
    if not latam_high_scope.empty:
        tmp = latam_high_scope[
            [
                "grado_num",
                "Prueba_filtro",
                "Nivel_EFA_filtro",
            ]
        ].drop_duplicates()
        tmp["Fuente"] = "LATAM"
        evidence_parts.append(tmp)

    if evidence_parts:
        evidence = pd.concat(
            evidence_parts,
            ignore_index=True,
        ).drop_duplicates()

        evidence["Perfil"] = (
            evidence["grado_num"].astype(str)
            + "° · "
            + evidence["Nivel_EFA_filtro"].astype(str)
            + " · "
            + evidence["Prueba_filtro"].astype(str)
        )
        source_summary = (
            evidence.groupby(
                ["grado_num", "Fuente"],
                as_index=False,
            )
            .agg(
                Cohortes=("Perfil", "nunique")
            )
        )
        pivot = source_summary.pivot(
            index="grado_num",
            columns="Fuente",
            values="Cohortes",
        ).fillna(0)

        st.markdown("### Cobertura de fuentes")
        fig_sources = px.imshow(
            pivot,
            text_auto=True,
            aspect="auto",
            labels={
                "x": "Fuente",
                "y": "Grado",
                "color": "Cohortes",
            },
            title=(
                "Número de cohortes con evidencia disponible "
                "por grado y fuente"
            ),
        )
        fig_sources.update_yaxes(
            tickmode="array",
            tickvals=list(pivot.index),
            ticktext=[f"{int(g)}°" for g in pivot.index],
        )
        fig_sources.update_layout(
            height=max(360, 48 * len(pivot.index) + 180),
        )
        st.plotly_chart(
            fig_sources,
            use_container_width=True,
            key="story_overview_sources",
        )

    st.markdown("### Evaluar para Avanzar · todas las cohortes")
    efa_catalog = catalog[catalog["EFA_ancla"].notna()].copy()
    if not efa_catalog.empty:
        fig_efa = px.scatter(
            efa_catalog,
            x="grado_num",
            y="EFA_ancla",
            color="Nivel_EFA_filtro",
            size="Grupo_n",
            facet_col="Prueba_filtro",
            facet_col_wrap=2,
            custom_data=[
                "Grupo_n",
                "Fuentes_maximas",
                "Fuentes_requeridas",
            ],
            category_orders={
                "Nivel_EFA_filtro": LEVEL_ORDER,
                "Prueba_filtro": IMPROVE_EFA_TESTS,
            },
            title=(
                "Puntaje EFA de cada cohorte con máxima "
                "cobertura de fuentes"
            ),
        )
        for y in [25, 50, 75]:
            fig_efa.add_hline(
                y=y,
                line_dash="dot",
                opacity=0.25,
            )
        fig_efa.update_xaxes(
            dtick=1,
            title="Grado",
        )
        fig_efa.update_yaxes(
            range=[0, 100],
            title="Puntaje EFA (%)",
        )
        fig_efa.update_traces(
            hovertemplate=(
                "Grado: %{x}°<br>"
                "EFA: %{y:.1f}%<br>"
                "n cohorte: %{customdata[0]}<br>"
                "Fuentes máximas: %{customdata[1]}<br>"
                "%{customdata[2]}"
                "<extra></extra>"
            )
        )
        fig_efa.for_each_annotation(
            lambda a: a.update(
                text=a.text.replace(
                    "Prueba_filtro=", ""
                )
            )
        )
        fig_efa.update_layout(
            height=780,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Nivel EFA",
        )
        st.plotly_chart(
            fig_efa,
            use_container_width=True,
            key="story_overview_efa",
        )

        catalog_table = efa_catalog.copy()
        catalog_table["Grado"] = catalog_table[
            "grado_num"
        ].map(lambda x: f"{int(x)}°")
        catalog_table["EFA"] = catalog_table[
            "EFA_ancla"
        ].map(lambda x: f"{x:.1f}%")
        st.markdown("#### Cohortes que sostienen el panorama")
        st.dataframe(
            catalog_table[
                [
                    "Grado",
                    "Nivel_EFA_filtro",
                    "Prueba_filtro",
                    "Grupo_n",
                    "EFA",
                    "Fuentes_maximas",
                    "Fuentes_requeridas",
                ]
            ].rename(
                columns={
                    "Nivel_EFA_filtro": "Nivel EFA",
                    "Prueba_filtro": "Prueba EFA",
                    "Grupo_n": "Estudiantes",
                    "Fuentes_maximas": "Fuentes máximas",
                    "Fuentes_requeridas": "Fuentes requeridas",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    internal_scope = _filter_story_scope(
        story_internal,
        selected_grade,
        selected_level,
        selected_test,
    )
    if not internal_scope.empty:
        st.markdown("### Evaluación interna · T1 → T2")
        internal_scope = internal_scope[
            internal_scope["Trimestre"]
            .astype(str)
            .isin(["TRIMESTRE 1", "TRIMESTRE 2"])
        ].copy()
        internal_scope["Trimestre"] = pd.Categorical(
            internal_scope["Trimestre"],
            categories=["TRIMESTRE 1", "TRIMESTRE 2"],
            ordered=True,
        )
        fig_internal = px.line(
            internal_scope.sort_values(
                [
                    "Prueba_filtro",
                    "Nivel_EFA_filtro",
                    "Trimestre",
                    "grado_num",
                ]
            ),
            x="grado_num",
            y="Promedio_interno",
            color="Nivel_EFA_filtro",
            line_dash="Trimestre",
            markers=True,
            facet_col="Prueba_filtro",
            facet_col_wrap=2,
            custom_data=["Estudiantes"],
            category_orders={
                "Nivel_EFA_filtro": LEVEL_ORDER,
                "Prueba_filtro": IMPROVE_EFA_TESTS,
            },
            title=(
                "Desempeño interno promedio de las mismas "
                "cohortes · Trimestre 1 y 2"
            ),
        )
        fig_internal.update_xaxes(
            dtick=1,
            title="Grado",
        )
        fig_internal.update_yaxes(
            range=[1, 4.15],
            title="Nivel interno promedio",
            tickmode="array",
            tickvals=[1, 2, 3, 4],
            ticktext=[
                "Progreso limitado",
                "Emergente",
                "En aceleración",
                "Avanzado",
            ],
        )
        fig_internal.update_traces(
            hovertemplate=(
                "Grado: %{x}°<br>"
                "Desempeño interno: %{y:.2f}<br>"
                "n: %{customdata[0]}"
                "<extra></extra>"
            )
        )
        fig_internal.for_each_annotation(
            lambda a: a.update(
                text=a.text.replace(
                    "Prueba_filtro=", ""
                )
            )
        )
        fig_internal.update_layout(
            height=820,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(
            fig_internal,
            use_container_width=True,
            key="story_overview_internal",
        )

    innovamat = low_scope[
        low_scope["Fuente"].astype(str) == "Innovamat"
    ].copy() if not low_scope.empty else pd.DataFrame()

    if not innovamat.empty:
        st.markdown("### Innovamat · todas las dimensiones disponibles")
        innovamat["Valor"] = pd.to_numeric(
            innovamat["Valor"], errors="coerce"
        )
        fig_innovamat = px.scatter(
            innovamat,
            x="grado_num",
            y="Valor",
            color="Nivel_EFA_filtro",
            facet_col="Resultado",
            facet_col_wrap=3,
            custom_data=["n", "Prueba_filtro"],
            category_orders={
                "Nivel_EFA_filtro": LEVEL_ORDER,
            },
            title=(
                "Resultados de Innovamat por cohorte, "
                "grado y dimensión"
            ),
        )
        fig_innovamat.update_xaxes(
            dtick=1,
            title="Grado",
        )
        fig_innovamat.update_yaxes(
            range=[0, 100],
            title="Puntaje equivalente (0–100)",
        )
        fig_innovamat.update_traces(
            hovertemplate=(
                "Grado: %{x}°<br>"
                "Resultado: %{y:.1f}<br>"
                "n: %{customdata[0]}<br>"
                "Cohorte definida por: %{customdata[1]}"
                "<extra></extra>"
            )
        )
        fig_innovamat.for_each_annotation(
            lambda a: a.update(
                text=a.text.replace(
                    "Resultado=", ""
                )
            )
        )
        fig_innovamat.update_layout(
            height=760,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Nivel EFA",
        )
        st.plotly_chart(
            fig_innovamat,
            use_container_width=True,
            key="story_overview_innovamat",
        )

    latam_parts = []
    if not low_scope.empty:
        latam_low = low_scope[
            low_scope["Fuente"].astype(str) == "LATAM"
        ].copy()
        if not latam_low.empty:
            latam_parts.append(latam_low)

    if not latam_high_scope.empty:
        latam_parts.append(latam_high_scope.copy())

    if latam_parts:
        latam = pd.concat(
            latam_parts,
            ignore_index=True,
        )
        st.markdown("### LATAM · niveles predominantes")
        latam["Valor"] = pd.to_numeric(
            latam["Valor"], errors="coerce"
        )
        fig_latam = px.scatter(
            latam,
            x="grado_num",
            y="Valor",
            color="Nivel_EFA_filtro",
            symbol="Resultado",
            facet_col="Prueba_filtro",
            facet_col_wrap=2,
            custom_data=["n", "Etiqueta", "Extra"],
            category_orders={
                "Nivel_EFA_filtro": LEVEL_ORDER,
                "Prueba_filtro": IMPROVE_EFA_TESTS,
            },
            title=(
                "Nivel LATAM predominante en las cohortes "
                "con evidencia disponible"
            ),
        )
        fig_latam.update_xaxes(
            dtick=1,
            title="Grado",
        )
        fig_latam.update_yaxes(
            range=[0.5, 4.5],
            title="Nivel LATAM",
            tickmode="array",
            tickvals=[1, 2, 3, 4],
            ticktext=[
                "Inicio",
                "Básico",
                "Satisfactorio",
                "Avanzado",
            ],
        )
        fig_latam.update_traces(
            hovertemplate=(
                "Grado: %{x}°<br>"
                "Nivel LATAM: %{customdata[1]}<br>"
                "n: %{customdata[0]}<br>"
                "Proporción: %{customdata[2]:.1f}%"
                "<extra></extra>"
            )
        )
        fig_latam.for_each_annotation(
            lambda a: a.update(
                text=a.text.replace(
                    "Prueba_filtro=", ""
                )
            )
        )
        fig_latam.update_layout(
            height=760,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(
            fig_latam,
            use_container_width=True,
            key="story_overview_latam",
        )

    improve_scope = _filter_story_scope(
        profiles,
        selected_grade,
        selected_level,
        selected_test,
    )
    if not improve_scope.empty:
        st.markdown("### Improve · crecimiento de las cohortes 10°–11°")
        improve_rows = []
        metric_map = {
            "Global": "Puntaje global",
            "Matemáticas": "Puntaje Matemáticas",
            "Lenguaje": "Puntaje Lectura crítica",
            "Ciencias naturales": "Puntaje Ciencias naturales",
            "Ciencias sociales": "Puntaje Sociales y ciudadanas",
            "Inglés": "Puntaje Inglés",
        }

        for _, row in improve_scope.iterrows():
            grade = int(row["grado_num"])
            moments = [1, 2] if grade == 10 else [1, 2, 3]
            for result_name, raw_name in metric_map.items():
                for moment in moments:
                    col = (
                        f"Improve_EV{moment}_"
                        f"{raw_name}_mediana"
                    )
                    value = row.get(col, np.nan)
                    if pd.isna(value):
                        continue
                    improve_rows.append(
                        {
                            "grado_num": grade,
                            "Prueba_filtro": row["Prueba_filtro"],
                            "Nivel_EFA_filtro": row[
                                "Nivel_EFA_filtro"
                            ],
                            "Resultado": result_name,
                            "Momento": f"Evaluación {moment}",
                            "Orden": moment,
                            "Valor": float(value),
                            "n": int(
                                row["Candidatos_completos"]
                            ),
                        }
                    )

        improve_long = pd.DataFrame(improve_rows)
        if not improve_long.empty:
            global_imp = improve_long[
                improve_long["Resultado"] == "Global"
            ].copy()
            fig_imp = px.line(
                global_imp.sort_values(
                    [
                        "grado_num",
                        "Prueba_filtro",
                        "Nivel_EFA_filtro",
                        "Orden",
                    ]
                ),
                x="Momento",
                y="Valor",
                color="Nivel_EFA_filtro",
                line_dash="Prueba_filtro",
                facet_col="grado_num",
                markers=True,
                custom_data=["n"],
                category_orders={
                    "Nivel_EFA_filtro": LEVEL_ORDER,
                },
                title=(
                    "Trayectoria global de Improve para todas "
                    "las cohortes disponibles"
                ),
            )
            fig_imp.update_yaxes(
                title="Puntaje global Improve",
                rangemode="tozero",
            )
            fig_imp.update_traces(
                hovertemplate=(
                    "Momento: %{x}<br>"
                    "Puntaje global: %{y:.1f}<br>"
                    "n: %{customdata[0]}"
                    "<extra></extra>"
                )
            )
            fig_imp.for_each_annotation(
                lambda a: a.update(
                    text=a.text.replace(
                        "grado_num=", "Grado "
                    )
                    + "°"
                )
            )
            fig_imp.update_layout(
                height=520,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
            )
            st.plotly_chart(
                fig_imp,
                use_container_width=True,
                key="story_overview_improve_global",
            )

            st.caption(
                "La tabla siguiente conserva todas las pruebas de Improve "
                "y todos los momentos disponibles para las cohortes "
                "seleccionadas."
            )
            table = improve_long.copy()
            table["Grado"] = table["grado_num"].map(
                lambda x: f"{int(x)}°"
            )
            table["Valor"] = table["Valor"].map(
                lambda x: f"{x:.1f}"
            )
            st.dataframe(
                table[
                    [
                        "Grado",
                        "Nivel_EFA_filtro",
                        "Prueba_filtro",
                        "Resultado",
                        "Momento",
                        "n",
                        "Valor",
                    ]
                ].rename(
                    columns={
                        "Nivel_EFA_filtro": "Nivel EFA",
                        "Prueba_filtro": "Prueba EFA que define la cohorte",
                        "Resultado": "Resultado Improve",
                        "Momento": "Momento Improve",
                        "n": "Estudiantes",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "El panorama usa únicamente cohortes con máxima cobertura viable "
        "de fuentes y al menos 5 estudiantes. Las cohortes pueden "
        "superponerse entre filtros, por eso no se suman como si fueran "
        "estudiantes distintos."
    )


def render_student_story(
    profiles: pd.DataFrame,
    profiles_3_9: pd.DataFrame,
    story_internal: pd.DataFrame,
    story_latam_10_11: pd.DataFrame,
):
    st.markdown("## Historia de un perfil representativo")
    st.caption(
        "Puedes seleccionar una combinación específica o elegir Todos "
        "para desplegar el panorama completo de evidencias disponibles."
    )

    if profiles.empty and profiles_3_9.empty:
        st.warning(
            "No está disponible la base de perfiles representativos."
        )
        return

    st.info(
        "La vista incluye únicamente estudiantes con la máxima cantidad "
        "viable de fuentes para su grado. Se exige un mínimo de 5 estudiantes "
        "para sostener una combinación. Así evitamos construir historias a "
        "partir de casos aislados y priorizamos los recorridos más completos."
    )

    catalog = _story_profile_catalog(
        profiles,
        profiles_3_9,
    )
    if catalog.empty:
        st.warning(
            "No hay cohortes con máxima cobertura de fuentes."
        )
        return

    grades = sorted(
        catalog["grado_num"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_grade = st.selectbox(
            "Grado",
            ["Todos"] + grades,
            format_func=lambda g: (
                "Todos los grados"
                if g == "Todos"
                else f"{g}°"
            ),
            key="story_grade",
        )

    key_scope = catalog.copy()
    if selected_grade != "Todos":
        key_scope = key_scope[
            key_scope["grado_num"] == int(selected_grade)
        ]

    available_levels = [
        level
        for level in LEVEL_ORDER
        if level in set(
            key_scope["Nivel_EFA_filtro"].astype(str)
        )
    ]
    with c2:
        selected_level = st.selectbox(
            "Nivel en Evaluar para Avanzar",
            ["Todos"] + available_levels,
            format_func=lambda level: (
                "Todos los niveles"
                if level == "Todos"
                else level
            ),
            key="story_level",
        )

    if selected_level != "Todos":
        key_scope = key_scope[
            key_scope["Nivel_EFA_filtro"].astype(str)
            == selected_level
        ]

    available_tests = [
        test
        for test in IMPROVE_EFA_TESTS
        if test in set(
            key_scope["Prueba_filtro"].astype(str)
        )
    ]
    with c3:
        selected_test = st.selectbox(
            "Prueba de Evaluar para Avanzar",
            ["Todas"] + available_tests,
            format_func=lambda test: (
                "Todas las pruebas"
                if test == "Todas"
                else test
            ),
            key="story_test",
        )

    broad_mode = (
        selected_grade == "Todos"
        or selected_level == "Todos"
        or selected_test == "Todas"
    )
    if broad_mode:
        _render_story_overview(
            profiles,
            profiles_3_9,
            story_internal,
            story_latam_10_11,
            selected_grade,
            selected_level,
            selected_test,
        )
        return

    if selected_grade <= 9:
        grade_data = profiles_3_9[
            profiles_3_9["grado_num"] == selected_grade
        ].copy()

        level_data = grade_data[
            grade_data["Nivel_EFA_filtro"].astype(str)
            == selected_level
        ].copy()

        d = level_data[
            level_data["Prueba_filtro"].astype(str)
            == selected_test
        ].copy()

        if d.empty:
            st.warning(
                "No hay un grupo representativo para esta combinación."
            )
            return

        group_n = int(d["Grupo_EFA_n"].max())
        efa_selected = d[
            (d["Fuente"].astype(str) == "Evaluar para Avanzar")
            & (d["Resultado"].astype(str) == selected_test)
        ].copy()
        anchor_score = (
            float(efa_selected.iloc[0]["Valor"])
            if not efa_selected.empty
            else np.nan
        )

        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Estudiantes en el grupo",
            f"{group_n}",
        )
        k2.metric(
            f"EFA · {selected_test}",
            (
                f"{anchor_score:.1f}%"
                if np.isfinite(anchor_score)
                else "—"
            ),
            delta=selected_level,
            delta_color="off",
        )
        max_sources = int(
            pd.to_numeric(
                d.get("Fuentes_maximas", pd.Series([0])),
                errors="coerce",
            ).fillna(0).max()
        )
        required_sources = str(
            d["Fuentes_requeridas"].iloc[0]
            if "Fuentes_requeridas" in d.columns
            else "Cobertura máxima de fuentes"
        )
        k3.metric(
            "Fuentes máximas del perfil",
            f"{max_sources}",
            help=required_sources,
        )
        st.caption(
            f"**Criterio de inclusión:** {required_sources}. "
            f"Solo se usan estudiantes que cumplen esta cobertura máxima "
            f"viable para {selected_grade}°."
        )

        st.markdown("### 1. Perfil en Evaluar para Avanzar")
        efa = d[
            d["Fuente"].astype(str) == "Evaluar para Avanzar"
        ].copy()
        if not efa.empty:
            efa["Selección"] = np.where(
                efa["Resultado"].astype(str) == selected_test,
                "Prueba seleccionada",
                "Otras pruebas",
            )
            efa["Nivel"] = efa["Valor"].map(_story_level_from_pct)

            fig_efa = px.bar(
                efa,
                x="Resultado",
                y="Valor",
                color="Selección",
                text=efa["Valor"].map(lambda x: f"{x:.1f}%"),
                custom_data=["n", "Nivel"],
                category_orders={
                    "Resultado": IMPROVE_EFA_TESTS,
                },
                title=(
                    f"Perfil EFA representativo · {selected_grade}° · "
                    f"{selected_level} en {selected_test}"
                ),
            )
            for y in [25, 50, 75]:
                fig_efa.add_hline(
                    y=y,
                    line_dash="dot",
                    opacity=0.35,
                )
            fig_efa.update_yaxes(
                range=[0, 100],
                title="Puntaje EFA (%)",
            )
            fig_efa.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "Prueba: %{x}<br>"
                    "Puntaje EFA: %{y:.1f}%<br>"
                    "n: %{customdata[0]}<br>"
                    "Nivel: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
            fig_efa.update_layout(
                height=470,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
            )
            st.plotly_chart(
                fig_efa,
                use_container_width=True,
                key="story_efa_profile_3_9",
            )

        _render_internal_story(
            story_internal,
            selected_grade,
            selected_test,
            selected_level,
            "story_internal_3_9",
        )

        st.markdown("### 2. Resultados adicionales disponibles")

        innovamat = d[
            d["Fuente"].astype(str) == "Innovamat"
        ].copy()
        if not innovamat.empty:
            st.markdown("#### Innovamat · perfil matemático")
            innovamat_order = [
                "Promedio general",
                "Numeración",
                "Relaciones y cambio",
                "Espacio y forma",
                "Medida",
                "Estadística y probabilidad",
            ]
            innovamat["Resultado"] = pd.Categorical(
                innovamat["Resultado"],
                categories=innovamat_order,
                ordered=True,
            )
            innovamat = innovamat.sort_values("Resultado")

            fig_inv = px.bar(
                innovamat,
                x="Resultado",
                y="Valor",
                text=innovamat["Valor"].map(
                    lambda x: f"{x:.1f}"
                ),
                custom_data=["n"],
                category_orders={
                    "Resultado": innovamat_order,
                },
                title=(
                    "Desempeño representativo en la actividad "
                    "matemática adicional"
                ),
            )
            fig_inv.update_yaxes(
                range=[0, 100],
                title="Puntaje equivalente (0–100)",
            )
            fig_inv.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "Dimensión: %{x}<br>"
                    "Resultado: %{y:.1f}/100<br>"
                    "n: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )
            fig_inv.update_layout(
                height=480,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_inv,
                use_container_width=True,
                key="story_innovamat_3_9",
            )

        latam = d[
            d["Fuente"].astype(str) == "LATAM"
        ].copy()
        if not latam.empty:
            st.markdown("#### LATAM · nivel de logro")
            latam_order = [
                "Matemáticas",
                "Lenguaje",
            ]
            latam["Resultado"] = pd.Categorical(
                latam["Resultado"],
                categories=latam_order,
                ordered=True,
            )
            latam = latam.sort_values("Resultado")

            fig_latam = px.bar(
                latam,
                x="Resultado",
                y="Valor",
                text=latam["Etiqueta"],
                custom_data=["n", "Extra"],
                category_orders={
                    "Resultado": latam_order,
                },
                title="Nivel de logro LATAM predominante",
            )
            fig_latam.update_yaxes(
                range=[0, 4.5],
                title="Nivel LATAM (1–4)",
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=[
                    "Inicio",
                    "Básico",
                    "Satisfactorio",
                    "Avanzado",
                ],
            )
            fig_latam.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "Área: %{x}<br>"
                    "Nivel predominante: %{text}<br>"
                    "n: %{customdata[0]}<br>"
                    "Proporción en ese nivel: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
            fig_latam.update_layout(
                height=420,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_latam,
                use_container_width=True,
                key="story_latam_3_9",
            )

        if innovamat.empty and latam.empty:
            st.caption(
                "Para esta combinación no hay otra fuente con al menos "
                "5 estudiantes emparejados además de EFA."
            )

        st.markdown("### 3. La historia en una frase")
        extra_bits = []
        if not innovamat.empty:
            inv_general = innovamat[
                innovamat["Resultado"].astype(str)
                == "Promedio general"
            ]
            if not inv_general.empty:
                extra_bits.append(
                    "en Innovamat el promedio general representativo es "
                    f"**{float(inv_general.iloc[0]['Valor']):.1f}/100**"
                )
        if not latam.empty:
            latam_txt = ", ".join(
                f"{str(r['Resultado'])}: **{r['Etiqueta']}**"
                for _, r in latam.iterrows()
            )
            extra_bits.append(
                f"en LATAM predominan {latam_txt}"
            )

        story_extra = (
            "; ".join(extra_bits)
            if extra_bits
            else "no hay otra evaluación con muestra suficiente"
        )
        st.success(
            f"En {selected_grade}°, este perfil representa a estudiantes "
            f"ubicados en **{selected_level} de {selected_test}** en EFA"
            + (
                f", con una mediana de **{anchor_score:.1f}%**. "
                if np.isfinite(anchor_score)
                else ". "
            )
            + story_extra
            + "."
        )

        st.caption(
            "De 3° a 9° no se incorpora Improve. Cada resultado adicional "
            "solo aparece si hay al menos 5 estudiantes emparejados dentro "
            "del grupo seleccionado."
        )
        return

    # 10° y 11° mantienen la historia longitudinal con Improve.
    grade_data = profiles[
        profiles["grado_num"] == selected_grade
    ].copy()

    level_data = grade_data[
        grade_data["Nivel_EFA_filtro"].astype(str)
        == selected_level
    ].copy()

    d = level_data[
        level_data["Prueba_filtro"].astype(str)
        == selected_test
    ].copy()

    if d.empty:
        st.warning(
            "No hay un grupo con recorrido completo para esta combinación."
        )
        return

    row = d.iloc[0]
    group_n = int(row["Candidatos_completos"])
    anchor_score = float(row["EFA_seleccionada_mediana"])

    final_ev = 2 if selected_grade == 10 else 3
    final_label = f"Evaluación {final_ev}"

    global_initial = float(
        row["Improve_EV1_Puntaje global_mediana"]
    )
    global_final = float(
        row[f"Improve_EV{final_ev}_Puntaje global_mediana"]
    )
    global_change = global_final - global_initial

    max_sources = int(
        pd.to_numeric(
            pd.Series([row.get("Fuentes_maximas", 0)]),
            errors="coerce",
        ).fillna(0).iloc[0]
    )
    required_sources = str(
        row.get(
            "Fuentes_requeridas",
            "Cobertura máxima de fuentes",
        )
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Grupo con recorrido completo",
        f"{group_n}",
        help=(
            "Estudiantes con máxima cobertura de fuentes y recorrido "
            "completo para construir este perfil."
        ),
    )
    k2.metric(
        "Fuentes máximas",
        f"{max_sources}",
        help=required_sources,
    )
    k3.metric(
        f"EFA · {selected_test}",
        f"{anchor_score:.1f}%",
        delta=selected_level,
        delta_color="off",
    )
    k4.metric(
        "Improve global · inicio",
        f"{global_initial:.1f}",
    )
    k5.metric(
        f"Improve global · {final_label.lower()}",
        f"{global_final:.1f}",
        delta=f"{global_change:+.1f}",
    )
    st.caption(
        f"**Criterio de inclusión:** {required_sources}. "
        f"Solo se usan estudiantes con esta cobertura máxima viable "
        f"para {selected_grade}°."
    )

    st.markdown("### 1. El perfil en Evaluar para Avanzar")

    efa_rows = []
    for test in IMPROVE_EFA_TESTS:
        value = row.get(f"EFA_{test}_mediana", np.nan)
        if pd.isna(value):
            continue
        efa_rows.append({
            "Prueba": test,
            "Puntaje EFA": float(value),
            "Nivel": _story_level_from_pct(value),
            "Selección": (
                "Prueba seleccionada"
                if test == selected_test
                else "Otras pruebas"
            ),
        })

    efa_profile = pd.DataFrame(efa_rows)

    if not efa_profile.empty:
        fig_efa = px.bar(
            efa_profile,
            x="Prueba",
            y="Puntaje EFA",
            color="Selección",
            text=efa_profile["Puntaje EFA"].map(
                lambda x: f"{x:.1f}%"
            ),
            custom_data=["Nivel"],
            title=(
                f"Perfil EFA representativo · {selected_grade}° · "
                f"{selected_level} en {selected_test}"
            ),
        )
        for y in [25, 50, 75]:
            fig_efa.add_hline(
                y=y,
                line_dash="dot",
                opacity=0.35,
            )
        fig_efa.update_yaxes(
            range=[0, 100],
            title="Puntaje EFA (%)",
        )
        fig_efa.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Prueba: %{x}<br>"
                "Puntaje EFA: %{y:.1f}%<br>"
                "Nivel: %{customdata[0]}"
                "<extra></extra>"
            ),
        )
        fig_efa.update_layout(
            height=480,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(
            fig_efa,
            use_container_width=True,
            key="story_efa_profile",
        )

    _render_internal_story(
        story_internal,
        selected_grade,
        selected_test,
        selected_level,
        "story_internal_10_11",
    )

    latam_story = story_latam_10_11[
        (story_latam_10_11["grado_num"] == selected_grade)
        & (
            story_latam_10_11["Prueba_filtro"].astype(str)
            == selected_test
        )
        & (
            story_latam_10_11["Nivel_EFA_filtro"].astype(str)
            == selected_level
        )
    ].copy() if (
        story_latam_10_11 is not None
        and not story_latam_10_11.empty
    ) else pd.DataFrame()

    if not latam_story.empty:
        st.markdown("### LATAM · misma cohorte de máxima cobertura")
        latam_order = ["Matemáticas", "Lenguaje"]
        latam_story["Resultado"] = pd.Categorical(
            latam_story["Resultado"],
            categories=latam_order,
            ordered=True,
        )
        latam_story = latam_story.sort_values("Resultado")
        fig_latam_story = px.bar(
            latam_story,
            x="Resultado",
            y="Valor",
            text="Etiqueta",
            custom_data=["n", "Extra"],
            category_orders={"Resultado": latam_order},
            title="Nivel LATAM predominante del mismo grupo",
        )
        fig_latam_story.update_yaxes(
            range=[0, 4.5],
            title="Nivel LATAM (1–4)",
            tickmode="array",
            tickvals=[1, 2, 3, 4],
            ticktext=[
                "Inicio",
                "Básico",
                "Satisfactorio",
                "Avanzado",
            ],
        )
        fig_latam_story.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Área: %{x}<br>"
                "Nivel predominante: %{text}<br>"
                "n: %{customdata[0]}<br>"
                "Proporción: %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        )
        fig_latam_story.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_latam_story,
            use_container_width=True,
            key="story_latam_10_11",
        )

    st.markdown("### 2. La trayectoria global en Improve")

    moments = [1, 2] if selected_grade == 10 else [1, 2, 3]
    global_rows = []
    for moment in moments:
        value = row.get(
            f"Improve_EV{moment}_Puntaje global_mediana",
            np.nan,
        )
        if pd.isna(value):
            continue
        global_rows.append({
            "Momento": f"Evaluación {moment}",
            "Orden": moment,
            "Puntaje global": float(value),
        })

    global_traj = pd.DataFrame(global_rows)
    if not global_traj.empty:
        fig_global = px.line(
            global_traj.sort_values("Orden"),
            x="Momento",
            y="Puntaje global",
            markers=True,
            text=global_traj["Puntaje global"].map(
                lambda x: f"{x:.1f}"
            ),
            category_orders={
                "Momento": [
                    f"Evaluación {m}" for m in moments
                ]
            },
            title="Cómo cambia el puntaje global de Improve",
        )
        fig_global.update_traces(
            textposition="top center",
            cliponaxis=False,
        )
        fig_global.update_yaxes(
            rangemode="tozero",
            title="Puntaje global Improve",
        )
        fig_global.update_layout(
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_global,
            use_container_width=True,
            key="story_improve_global",
        )

    st.markdown("### 3. Qué ocurre en cada prueba de Improve")

    improve_map = {
        "Matemáticas": "Puntaje Matemáticas",
        "Lenguaje": "Puntaje Lectura crítica",
        "Ciencias naturales": "Puntaje Ciencias naturales",
        "Ciencias sociales": "Puntaje Sociales y ciudadanas",
        "Inglés": "Puntaje Inglés",
    }

    subject_rows = []
    for test, raw_name in improve_map.items():
        for moment in moments:
            value = row.get(
                f"Improve_EV{moment}_{raw_name}_mediana",
                np.nan,
            )
            if pd.isna(value):
                continue
            subject_rows.append({
                "Prueba": test,
                "Momento": f"Evaluación {moment}",
                "Orden": moment,
                "Promedio Improve": float(value),
            })

    subject_traj = pd.DataFrame(subject_rows)

    if not subject_traj.empty:
        fig_subject = px.line(
            subject_traj.sort_values(["Prueba", "Orden"]),
            x="Momento",
            y="Promedio Improve",
            color="Prueba",
            markers=True,
            category_orders={
                "Momento": [
                    f"Evaluación {m}" for m in moments
                ],
                "Prueba": IMPROVE_EFA_TESTS,
            },
            title="Recorrido por las cinco pruebas de Improve",
        )
        fig_subject.update_yaxes(
            range=[0, 100],
            title="Puntaje Improve",
        )
        fig_subject.update_xaxes(title=None)
        fig_subject.update_layout(
            height=540,
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Prueba",
        )
        st.plotly_chart(
            fig_subject,
            use_container_width=True,
            key="story_improve_subjects",
        )

        table_rows = []
        for test, raw_name in improve_map.items():
            efa_value = row.get(
                f"EFA_{test}_mediana",
                np.nan,
            )
            initial = row.get(
                f"Improve_EV1_{raw_name}_mediana",
                np.nan,
            )
            final = row.get(
                f"Improve_EV{final_ev}_{raw_name}_mediana",
                np.nan,
            )
            if pd.isna(initial) or pd.isna(final):
                continue
            table_rows.append({
                "Prueba": test,
                "EFA": float(efa_value)
                if not pd.isna(efa_value)
                else np.nan,
                "Nivel EFA": _story_level_from_pct(efa_value),
                "Improve inicial": float(initial),
                "Improve final": float(final),
                "Cambio": float(final) - float(initial),
            })

        story_table = pd.DataFrame(table_rows)

        if not story_table.empty:
            display = story_table.copy()
            display["EFA"] = display["EFA"].map(
                lambda x: f"{x:.1f}%"
            )
            display["Improve inicial"] = display[
                "Improve inicial"
            ].map(lambda x: f"{x:.1f}")
            display["Improve final"] = display[
                "Improve final"
            ].map(lambda x: f"{x:.1f}")
            display["Cambio"] = display["Cambio"].map(
                lambda x: f"{x:+.1f}"
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

            best = story_table.sort_values(
                "Cambio",
                ascending=False,
            ).iloc[0]
            worst = story_table.sort_values(
                "Cambio",
                ascending=True,
            ).iloc[0]

            st.markdown("### 4. La historia en una frase")
            direction = (
                "aumenta"
                if global_change > 0
                else "disminuye"
                if global_change < 0
                else "se mantiene"
            )
            st.success(
                f"Este perfil representativo parte de **{selected_level} en "
                f"{selected_test}** ({anchor_score:.1f}% en EFA). "
                f"En Improve, su puntaje global {direction} de "
                f"**{global_initial:.1f}** a **{global_final:.1f}** "
                f"({global_change:+.1f} puntos). La mayor mejora por prueba "
                f"se observa en **{best['Prueba']}** "
                f"({best['Cambio']:+.1f}), mientras que el cambio más bajo "
                f"se presenta en **{worst['Prueba']}** "
                f"({worst['Cambio']:+.1f})."
            )

    st.caption(
        "El perfil representa el comportamiento mediano de un grupo con "
        "recorrido completo; no corresponde al expediente de una persona "
        "identificable."
    )
