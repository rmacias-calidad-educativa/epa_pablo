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


def _evolution_filter(
    data: pd.DataFrame,
    metric: str | None = None,
    selected_grades: list[int] | None = None,
    selected_sites: list[str] | None = None,
) -> pd.DataFrame:
    d = data.copy()
    if metric is not None and "Metrica" in d.columns:
        d = d[d["Metrica"].astype(str) == metric]
    if selected_grades:
        d = d[d["Grado_num"].isin(selected_grades)]
    if selected_sites and "Sede" in d.columns:
        d = d[d["Sede"].astype(str).isin(selected_sites)]
    return d.copy()


def _evolution_combine(data: pd.DataFrame) -> dict:
    if data is None or data.empty:
        return {
            "n": 0,
            "initial": np.nan,
            "final": np.nan,
            "change": np.nan,
            "improved": np.nan,
            "same": np.nan,
            "decreased": np.nan,
        }

    n = pd.to_numeric(data["N"], errors="coerce").fillna(0).sum()
    if n <= 0:
        return {
            "n": 0,
            "initial": np.nan,
            "final": np.nan,
            "change": np.nan,
            "improved": np.nan,
            "same": np.nan,
            "decreased": np.nan,
        }

    sum_initial = pd.to_numeric(
        data["Sum_Inicial"], errors="coerce"
    ).fillna(0).sum()
    sum_final = pd.to_numeric(
        data["Sum_Final"], errors="coerce"
    ).fillna(0).sum()
    sum_change = pd.to_numeric(
        data["Sum_Cambio"], errors="coerce"
    ).fillna(0).sum()
    n_improved = pd.to_numeric(
        data["N_Mejora"], errors="coerce"
    ).fillna(0).sum()
    n_same = pd.to_numeric(
        data["N_Igual"], errors="coerce"
    ).fillna(0).sum()
    n_down = pd.to_numeric(
        data["N_Baja"], errors="coerce"
    ).fillna(0).sum()

    return {
        "n": int(n),
        "initial": sum_initial / n,
        "final": sum_final / n,
        "change": sum_change / n,
        "improved": n_improved / n * 100,
        "same": n_same / n * 100,
        "decreased": n_down / n * 100,
    }


def _evolution_grouped(
    data: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    if data.empty:
        return pd.DataFrame()

    for keys, group in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        info = _evolution_combine(group)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "N": info["n"],
            "Promedio inicial": info["initial"],
            "Promedio final": info["final"],
            "Cambio promedio": info["change"],
            "% mejora": info["improved"],
            "% igual": info["same"],
            "% baja": info["decreased"],
        })
        rows.append(row)

    return pd.DataFrame(rows)


def _observed_value(
    observed: pd.DataFrame,
    grade: int,
    site: str,
    moment: str,
    metric: str,
) -> dict:
    d = observed[
        (observed["Grado_num"] == grade)
        & (observed["Sede"].astype(str) == site)
        & (observed["Momento"].astype(str) == moment)
        & (observed["Metrica"].astype(str) == metric)
    ].copy()
    if d.empty:
        return {"n": 0, "mean": np.nan}
    row = d.iloc[0]
    return {
        "n": int(pd.to_numeric(row["N"], errors="coerce")),
        "mean": float(pd.to_numeric(row["Promedio"], errors="coerce")),
    }


def _paired_site_table(
    summary: pd.DataFrame,
    grade: int,
    metric: str,
) -> pd.DataFrame:
    d = summary[
        (summary["Grado_num"] == grade)
        & (summary["Metrica"].astype(str) == metric)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for _, r in d.iterrows():
        n = float(r["N"])
        if n <= 0:
            continue
        initial = float(r["Sum_Inicial"]) / n
        final = float(r["Sum_Final"]) / n
        change = float(r["Sum_Cambio"]) / n
        improved = float(r["N_Mejora"]) / n * 100
        rows.append({
            "Sede": r["Sede"],
            "n pareados": int(n),
            "Primera aplicación": initial,
            "Última aplicación": final,
            "Cambio pareado": change,
            "% mejora": improved,
        })
    return pd.DataFrame(rows)


def render_evolution_10_11(
    summary: pd.DataFrame,
    trajectory: pd.DataFrame,
    availability: pd.DataFrame,
    observed: pd.DataFrame,
):
    st.markdown("## Evolución de resultados · 10° y 11°")
    st.caption(
        "Evolución de puntajes entre aplicaciones, con comparación "
        "pareada de los mismos estudiantes."
    )

    if summary.empty or observed.empty:
        st.warning("No está disponible la información de evolución.")
        return

    st.info(
        "El crecimiento se calcula comparando a los mismos estudiantes "
        "entre aplicaciones. Cota se muestra cuando tiene información, "
        "aunque no tenga una segunda aplicación para calcular cambio."
    )

    tab10, tab11 = st.tabs(
        ["10° · Crecimiento", "11° · Mosquera"]
    )

    with tab10:
        st.markdown("### Crecimiento global · mismos estudiantes")
        st.caption(
            "Compara Evaluación 1 vs Evaluación 2 únicamente en estudiantes "
            "que presentaron ambas aplicaciones."
        )

        paired10 = _paired_site_table(
            summary,
            10,
            "Puntaje global",
        )

        if not paired10.empty:
            overall10 = _evolution_combine(
                summary[
                    (summary["Grado_num"] == 10)
                    & (
                        summary["Metrica"].astype(str)
                        == "Puntaje global"
                    )
                ]
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Estudiantes pareados",
                f"{overall10['n']:,}".replace(",", "."),
            )
            k2.metric(
                "Primera aplicación · pareados",
                f"{overall10['initial']:.1f}",
            )
            k3.metric(
                "Segunda aplicación · pareados",
                f"{overall10['final']:.1f}",
            )
            k4.metric(
                "Crecimiento promedio",
                f"{overall10['change']:+.1f}",
            )

            long10 = pd.concat(
                [
                    paired10[
                        ["Sede", "Primera aplicación"]
                    ]
                    .rename(
                        columns={"Primera aplicación": "Promedio"}
                    )
                    .assign(Evaluación="Evaluación 1"),
                    paired10[
                        ["Sede", "Última aplicación"]
                    ]
                    .rename(
                        columns={"Última aplicación": "Promedio"}
                    )
                    .assign(Evaluación="Evaluación 2"),
                ],
                ignore_index=True,
            )

            # Cota solo cuenta con Evaluación 1. Se muestra para dar
            # visibilidad a la sede, pero no entra al cálculo de crecimiento.
            cota_ev1 = _observed_value(
                observed,
                10,
                "Cota",
                "Evaluación 1",
                "Puntaje global",
            )
            if cota_ev1["n"] > 0 and np.isfinite(cota_ev1["mean"]):
                long10 = pd.concat(
                    [
                        long10,
                        pd.DataFrame([{
                            "Sede": "Cota",
                            "Promedio": cota_ev1["mean"],
                            "Evaluación": "Evaluación 1",
                        }]),
                    ],
                    ignore_index=True,
                )

            fig10_growth = px.bar(
                long10,
                x="Sede",
                y="Promedio",
                color="Evaluación",
                barmode="group",
                text=long10["Promedio"].map(
                    lambda x: f"{x:.1f}"
                ),
                title=(
                    "Puntaje global · Evaluación 1 vs Evaluación 2 "
                    "(Cota solo cuenta con Evaluación 1)"
                ),
                category_orders={
                    "Sede": ["Cota", "Mosquera", "Tunja", "Zipaquirá"]
                },
            )
            fig10_growth.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig10_growth.update_layout(
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Aplicación",
            )
            st.plotly_chart(
                fig10_growth,
                use_container_width=True,
                key="evolution10_paired_global",
            )

            growth_display = paired10.copy()
            for col in [
                "Primera aplicación",
                "Última aplicación",
            ]:
                growth_display[col] = growth_display[col].map(
                    lambda x: f"{x:.1f}"
                )
            growth_display["Cambio pareado"] = growth_display[
                "Cambio pareado"
            ].map(lambda x: f"{x:+.1f}")
            growth_display["% mejora"] = growth_display[
                "% mejora"
            ].map(lambda x: f"{x:.1f}%")

            cota_status = pd.DataFrame([{
                "Sede": "Cota",
                "n pareados": "—",
                "Primera aplicación": (
                    f"{cota_ev1['mean']:.1f}"
                    if cota_ev1["n"] > 0
                    and np.isfinite(cota_ev1["mean"])
                    else "Sin dato"
                ),
                "Última aplicación": "Sin aplicación 2",
                "Cambio pareado": "—",
                "% mejora": "—",
            }])
            growth_display = pd.concat(
                [growth_display, cota_status],
                ignore_index=True,
            )

            st.dataframe(
                growth_display[
                    [
                        "Sede",
                        "n pareados",
                        "Primera aplicación",
                        "Última aplicación",
                        "Cambio pareado",
                        "% mejora",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            best10 = paired10.sort_values(
                "Cambio pareado",
                ascending=False,
            ).iloc[0]
            st.success(
                f"Mayor crecimiento en 10°: **{best10['Sede']}**, "
                f"con {best10['Cambio pareado']:+.1f} puntos en promedio."
            )
            st.caption(
                "Cota aparece explícitamente, pero no tiene Evaluación 2; "
                "por eso no puede calcularse crecimiento pareado."
            )

        st.divider()
        st.markdown("### Comportamiento por cada prueba")
        st.caption(
            "Cada prueba compara Evaluación 1 vs Evaluación 2 por sede. "
            "El cambio se calcula únicamente con estudiantes que presentaron "
            "ambas aplicaciones."
        )

        subject_metrics_10 = [
            ("Matemáticas", "Puntaje Matemáticas"),
            ("Lenguaje", "Puntaje Lectura crítica"),
            ("Ciencias naturales", "Puntaje Ciencias naturales"),
            ("Ciencias sociales", "Puntaje Sociales y ciudadanas"),
            ("Inglés", "Puntaje Inglés"),
        ]

        subject_tabs_10 = st.tabs(
            [label for label, _ in subject_metrics_10]
        )

        for subject_tab, (subject_label, subject_metric) in zip(
            subject_tabs_10,
            subject_metrics_10,
        ):
            with subject_tab:
                paired_subject = _paired_site_table(
                    summary,
                    10,
                    subject_metric,
                )

                subject_overall = _evolution_combine(
                    summary[
                        (summary["Grado_num"] == 10)
                        & (
                            summary["Metrica"].astype(str)
                            == subject_metric
                        )
                    ]
                )

                s1, s2, s3, s4 = st.columns(4)
                s1.metric(
                    "Estudiantes pareados",
                    f"{subject_overall['n']:,}".replace(",", "."),
                )
                s2.metric(
                    "Evaluación 1 · pareados",
                    f"{subject_overall['initial']:.1f}",
                )
                s3.metric(
                    "Evaluación 2 · pareados",
                    f"{subject_overall['final']:.1f}",
                )
                s4.metric(
                    "Cambio promedio",
                    f"{subject_overall['change']:+.1f}",
                )

                if paired_subject.empty:
                    st.info(
                        "No hay estudiantes pareados para esta prueba."
                    )
                    continue

                subject_long = pd.concat(
                    [
                        paired_subject[
                            ["Sede", "Primera aplicación"]
                        ]
                        .rename(
                            columns={
                                "Primera aplicación": "Promedio"
                            }
                        )
                        .assign(Evaluación="Evaluación 1"),
                        paired_subject[
                            ["Sede", "Última aplicación"]
                        ]
                        .rename(
                            columns={
                                "Última aplicación": "Promedio"
                            }
                        )
                        .assign(Evaluación="Evaluación 2"),
                    ],
                    ignore_index=True,
                )

                cota_subject_ev1 = _observed_value(
                    observed,
                    10,
                    "Cota",
                    "Evaluación 1",
                    subject_metric,
                )
                if (
                    cota_subject_ev1["n"] > 0
                    and np.isfinite(cota_subject_ev1["mean"])
                ):
                    subject_long = pd.concat(
                        [
                            subject_long,
                            pd.DataFrame([{
                                "Sede": "Cota",
                                "Promedio": cota_subject_ev1["mean"],
                                "Evaluación": "Evaluación 1",
                            }]),
                        ],
                        ignore_index=True,
                    )

                fig_subject_10 = px.bar(
                    subject_long,
                    x="Sede",
                    y="Promedio",
                    color="Evaluación",
                    barmode="group",
                    text=subject_long["Promedio"].map(
                        lambda x: f"{x:.1f}"
                    ),
                    category_orders={
                        "Sede": [
                            "Cota",
                            "Mosquera",
                            "Tunja",
                            "Zipaquirá",
                        ]
                    },
                    title=(
                        f"{subject_label} · Evaluación 1 vs "
                        "Evaluación 2"
                    ),
                )
                fig_subject_10.update_yaxes(
                    range=[0, 100],
                    title="Puntaje promedio",
                )
                fig_subject_10.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )
                fig_subject_10.update_layout(
                    height=490,
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Aplicación",
                )
                st.plotly_chart(
                    fig_subject_10,
                    use_container_width=True,
                    key=(
                        "evolution10_subject_"
                        + subject_metric.replace(" ", "_")
                    ),
                )

                subject_display = paired_subject.copy()
                for col in [
                    "Primera aplicación",
                    "Última aplicación",
                ]:
                    subject_display[col] = subject_display[col].map(
                        lambda x: f"{x:.1f}"
                    )
                subject_display["Cambio pareado"] = subject_display[
                    "Cambio pareado"
                ].map(lambda x: f"{x:+.1f}")
                subject_display["% mejora"] = subject_display[
                    "% mejora"
                ].map(lambda x: f"{x:.1f}%")

                cota_subject_status = pd.DataFrame([{
                    "Sede": "Cota",
                    "n pareados": "—",
                    "Primera aplicación": (
                        f"{cota_subject_ev1['mean']:.1f}"
                        if cota_subject_ev1["n"] > 0
                        and np.isfinite(cota_subject_ev1["mean"])
                        else "Sin dato"
                    ),
                    "Última aplicación": "Sin aplicación 2",
                    "Cambio pareado": "—",
                    "% mejora": "—",
                }])

                subject_display = pd.concat(
                    [subject_display, cota_subject_status],
                    ignore_index=True,
                )

                st.dataframe(
                    subject_display[
                        [
                            "Sede",
                            "n pareados",
                            "Primera aplicación",
                            "Última aplicación",
                            "Cambio pareado",
                            "% mejora",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                ranked_subject = paired_subject.sort_values(
                    "Cambio pareado",
                    ascending=False,
                )
                if not ranked_subject.empty:
                    best_subject = ranked_subject.iloc[0]
                    st.caption(
                        f"Mayor avance en {subject_label}: "
                        f"**{best_subject['Sede']}** "
                        f"({best_subject['Cambio pareado']:+.1f} puntos)."
                    )

    with tab11:
        st.markdown("### Mosquera · crecimiento global")
        st.caption(
            "11° tiene tres aplicaciones. La trayectoria se calcula sobre los "
            "51 estudiantes que cuentan con Evaluación 1, 2 y 3."
        )

        pair11 = summary[
            (summary["Grado_num"] == 11)
            & (summary["Sede"].astype(str) == "Mosquera")
        ].copy()
        global11 = _evolution_combine(
            pair11[
                pair11["Metrica"].astype(str) == "Puntaje global"
            ]
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Estudiantes completos", f"{global11['n']}")
        c2.metric("Evaluación 1", f"{global11['initial']:.1f}")
        c3.metric("Evaluación 3", f"{global11['final']:.1f}")
        c4.metric("Crecimiento 1 → 3", f"{global11['change']:+.1f}")
        c5.metric("% que mejora", f"{global11['improved']:.1f}%")

        traj11_global = trajectory[
            (trajectory["Grado_num"] == 11)
            & (trajectory["Sede"].astype(str) == "Mosquera")
            & (
                trajectory["Metrica"].astype(str)
                == "Puntaje global"
            )
        ].copy()
        if not traj11_global.empty:
            traj11_global["Promedio"] = (
                traj11_global["Sum_Puntaje"]
                / traj11_global["N"]
            )
            traj11_global["Evaluación"] = (
                traj11_global["Orden"].map({
                    1: "Evaluación 1",
                    2: "Evaluación 2",
                    3: "Evaluación 3",
                })
            )
            fig11_global = px.bar(
                traj11_global.sort_values("Orden"),
                x="Evaluación",
                y="Promedio",
                text=traj11_global["Promedio"].map(
                    lambda x: f"{x:.1f}"
                ),
                title=(
                    "Puntaje global · mismos 51 estudiantes "
                    "en las tres aplicaciones"
                ),
            )
            fig11_global.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig11_global.update_layout(
                height=470,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig11_global,
                use_container_width=True,
                key="evolution11_global_three_apps",
            )

        st.divider()
        st.markdown("### Mosquera · crecimiento por prueba")

        subjects11 = pair11[
            pair11["Metrica"].astype(str) != "Puntaje global"
        ].copy()
        evolution_subject_labels = {
            "Puntaje Matemáticas": "Matemáticas",
            "Puntaje Lectura crítica": "Lenguaje",
            "Puntaje Ciencias naturales": "Ciencias naturales",
            "Puntaje Sociales y ciudadanas": "Ciencias sociales",
            "Puntaje Inglés": "Inglés",
        }
        subject_rows = []
        for _, r in subjects11.iterrows():
            n = float(r["N"])
            subject_rows.append({
                "Prueba": evolution_subject_labels.get(
                    str(r["Metrica"]),
                    str(r["Metrica"]).replace("Puntaje ", ""),
                ),
                "n": int(n),
                "Evaluación 1": float(r["Sum_Inicial"]) / n,
                "Evaluación 3": float(r["Sum_Final"]) / n,
                "Crecimiento 1 → 3": float(r["Sum_Cambio"]) / n,
                "% mejora": float(r["N_Mejora"]) / n * 100,
            })
        subject11 = pd.DataFrame(subject_rows)

        traj11_subjects = trajectory[
            (trajectory["Grado_num"] == 11)
            & (trajectory["Sede"].astype(str) == "Mosquera")
            & (
                trajectory["Metrica"].astype(str)
                != "Puntaje global"
            )
        ].copy()
        if not traj11_subjects.empty:
            traj11_subjects["Promedio"] = (
                traj11_subjects["Sum_Puntaje"]
                / traj11_subjects["N"]
            )
            traj11_subjects["Prueba"] = (
                traj11_subjects["Metrica"]
                .astype(str)
                .map(evolution_subject_labels)
                .fillna(
                    traj11_subjects["Metrica"]
                    .astype(str)
                    .str.replace("Puntaje ", "", regex=False)
                )
            )
            traj11_subjects["Evaluación"] = (
                traj11_subjects["Orden"].map({
                    1: "Evaluación 1",
                    2: "Evaluación 2",
                    3: "Evaluación 3",
                })
            )

            fig11_subjects = px.bar(
                traj11_subjects.sort_values(
                    ["Prueba", "Orden"]
                ),
                x="Prueba",
                y="Promedio",
                color="Evaluación",
                barmode="group",
                text=traj11_subjects["Promedio"].map(
                    lambda x: f"{x:.1f}"
                ),
                title=(
                    "Puntajes por prueba · Evaluación 1, 2 y 3 "
                    "(mismos 51 estudiantes)"
                ),
            )
            fig11_subjects.update_yaxes(
                range=[0, 100],
                title="Puntaje promedio",
            )
            fig11_subjects.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig11_subjects.update_layout(
                height=560,
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Aplicación",
            )
            st.plotly_chart(
                fig11_subjects,
                use_container_width=True,
                key="evolution11_subjects_three_apps",
            )

        if not subject11.empty:
            subject_display = subject11.copy()
            for col in ["Evaluación 1", "Evaluación 3"]:
                subject_display[col] = subject_display[col].map(
                    lambda x: f"{x:.1f}"
                )
            subject_display["Crecimiento 1 → 3"] = subject_display[
                "Crecimiento 1 → 3"
            ].map(lambda x: f"{x:+.1f}")
            subject_display["% mejora"] = subject_display[
                "% mejora"
            ].map(lambda x: f"{x:.1f}%")
            st.dataframe(
                subject_display,
                use_container_width=True,
                hide_index=True,
            )

            fig11_delta = px.bar(
                subject11.sort_values(
                    "Crecimiento 1 → 3",
                    ascending=False,
                ),
                x="Prueba",
                y="Crecimiento 1 → 3",
                text=subject11["Crecimiento 1 → 3"].map(
                    lambda x: f"{x:+.1f}"
                ),
                title=(
                    "Crecimiento promedio por prueba · "
                    "Evaluación 1 → Evaluación 3"
                ),
            )
            fig11_delta.add_hline(y=0, line_dash="dot")
            fig11_delta.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig11_delta.update_layout(
                height=470,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig11_delta,
                use_container_width=True,
                key="evolution11_subject_growth",
            )
