
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from processing import aggregate_group, build_attempt_table
from ui import (
    LEVEL_COLORS,
    LEVEL_ORDER,
    apply_filters,
    dataframe_download,
    multiselect_filter,
)


st.set_page_config(
    page_title="Estado de llegada | Evaluaciones",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: 12px 14px;
        border-radius: 14px;
    }
    .small-note {
        font-size: .90rem;
        color: rgba(120,120,120,.95);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def process_data(raw: pd.DataFrame) -> pd.DataFrame:
    return build_attempt_table(raw)


def pct(x):
    return f"{x:.1f}%" if pd.notna(x) else "—"


def num(x, digits=1):
    return f"{x:.{digits}f}" if pd.notna(x) else "—"


st.title("📊 Estado de llegada de los estudiantes")
st.caption(
    "Dashboard diagnóstico para identificar niveles de desempeño, cobertura de respuesta "
    "y concentraciones de reto por prueba, sede, grado, curso (cuando exista) y procedencia."
)

with st.expander("Reglas de cálculo", expanded=False):
    st.markdown(
        """
- **Progreso limitado:** 0% a 25% (incluye 25%).
- **Emergente:** >25% a 50%.
- **En aceleración:** >50% a 75%.
- **Avanzado:** >75% a 100%.
- **Denominador general:** 20 ítems.
- **Inglés 9° y 10°:** 22 ítems.
- **Inglés 11°:** 25 ítems.
- Si una pregunta no fue contestada y por eso no aparece en la exportación, **se conserva en el denominador esperado**.
        """
    )

uploaded = st.file_uploader(
    "Carga el archivo Excel exportado",
    type=["xlsx", "xls"],
    help="La aplicación procesa la base a nivel de ítem-respuesta.",
)

if uploaded is None:
    default_path = Path("data/default_summary.csv")
    if not default_path.exists():
        st.info("Carga el Excel para activar el dashboard.")
        st.stop()

    summary_data = pd.read_csv(default_path)
    st.success(
        "Mostrando la base precargada anonimizada. "
        "Puedes cargar el Excel completo arriba para habilitar el análisis individual."
    )

    st.sidebar.header("Filtros")
    sedes = st.sidebar.multiselect(
        "Sede",
        sorted(summary_data["Sede"].dropna().astype(str).unique()),
    )
    grados = st.sidebar.multiselect(
        "Grado",
        sorted(summary_data["Grado"].dropna().astype(str).unique()),
    )
    areas = st.sidebar.multiselect(
        "Área",
        sorted(summary_data["Area"].dropna().astype(str).unique()),
    )

    default_filtered = summary_data.copy()
    if sedes:
        default_filtered = default_filtered[default_filtered["Sede"].astype(str).isin(sedes)]
    if grados:
        default_filtered = default_filtered[default_filtered["Grado"].astype(str).isin(grados)]
    if areas:
        default_filtered = default_filtered[default_filtered["Area"].astype(str).isin(areas)]

    if default_filtered.empty:
        st.warning("Los filtros seleccionados no dejan registros para analizar.")
        st.stop()

    total_intentos = default_filtered["Intentos"].sum()
    weighted_avg = (
        (default_filtered["Promedio"] * default_filtered["Intentos"]).sum()
        / total_intentos
    )
    weighted_coverage = (
        (default_filtered["Cobertura"] * default_filtered["Intentos"]).sum()
        / total_intentos
    )
    support_pct = (
        (
            (default_filtered["Progreso_limitado"] + default_filtered["Emergente"])
            * default_filtered["Intentos"]
        ).sum()
        / total_intentos
    )
    sedes_n = default_filtered["Sede"].nunique()
    grados_n = default_filtered["Grado"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Intentos", f"{int(total_intentos):,}".replace(",", "."))
    k2.metric("Promedio", f"{weighted_avg:.1f}%")
    k3.metric("En ≤50%", f"{support_pct:.1f}%")
    k4.metric("Cobertura", f"{weighted_coverage:.1f}%")
    k5.metric("Sedes / grados", f"{sedes_n} / {grados_n}")

    t1, t2, t3 = st.tabs(["Panorama", "Cruce sede × grado", "Cruce área × grado"])

    with t1:
        level_totals = pd.DataFrame({
            "Nivel": LEVEL_ORDER,
            "Porcentaje": [
                (default_filtered["Progreso_limitado"] * default_filtered["Intentos"]).sum() / total_intentos,
                (default_filtered["Emergente"] * default_filtered["Intentos"]).sum() / total_intentos,
                (default_filtered["En_aceleracion"] * default_filtered["Intentos"]).sum() / total_intentos,
                (default_filtered["Avanzado"] * default_filtered["Intentos"]).sum() / total_intentos,
            ],
        })
        fig = px.bar(
            level_totals,
            x="Nivel",
            y="Porcentaje",
            color="Nivel",
            category_orders={"Nivel": LEVEL_ORDER},
            color_discrete_map=LEVEL_COLORS,
            text=level_totals["Porcentaje"].map(lambda x: f"{x:.1f}%"),
            title="Distribución global de niveles de desempeño",
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        fig.update_yaxes(range=[0, 100], title="% de intentos")
        st.plotly_chart(fig, use_container_width=True)

        heat = default_filtered.copy()
        heat["Requiere_apoyo"] = heat["Progreso_limitado"] + heat["Emergente"]
        heat["Grado_etiqueta"] = heat["Grado_num"].astype(int).astype(str) + "°"
        heat = (
            heat.groupby(["Sede", "Grado_etiqueta"], as_index=False)
            .apply(
                lambda g: pd.Series({
                    "Requiere_apoyo": (
                        (g["Requiere_apoyo"] * g["Intentos"]).sum()
                        / g["Intentos"].sum()
                    )
                }),
                include_groups=False,
            )
        )
        pivot = heat.pivot(
            index="Sede",
            columns="Grado_etiqueta",
            values="Requiere_apoyo",
        )
        cols = sorted(pivot.columns, key=lambda x: int(str(x).replace("°", "")))
        pivot = pivot[cols]
        fig = px.imshow(
            pivot,
            text_auto=".0f",
            aspect="auto",
            zmin=0,
            zmax=100,
            labels={
                "x": "Grado",
                "y": "Sede",
                "color": "% en Progreso limitado + Emergente",
            },
            title="Mapa de concentración de retos",
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        challenge = default_filtered.copy()
        challenge["Requiere_apoyo"] = challenge["Progreso_limitado"] + challenge["Emergente"]
        by_site = (
            challenge.groupby("Sede", as_index=False)
            .apply(
                lambda g: pd.Series({
                    "Intentos": g["Intentos"].sum(),
                    "Promedio": (g["Promedio"] * g["Intentos"]).sum() / g["Intentos"].sum(),
                    "Requiere_apoyo": (g["Requiere_apoyo"] * g["Intentos"]).sum() / g["Intentos"].sum(),
                }),
                include_groups=False,
            )
            .sort_values("Requiere_apoyo", ascending=False)
        )
        fig = px.bar(
            by_site,
            x="Sede",
            y="Requiere_apoyo",
            text=by_site["Requiere_apoyo"].map(lambda x: f"{x:.1f}%"),
            title="Concentración de retos por sede",
            labels={"Requiere_apoyo": "% en ≤50"},
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            default_filtered.sort_values(
                ["Progreso_limitado", "Emergente"],
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with t3:
        area_grade = default_filtered.copy()
        area_grade["Requiere_apoyo"] = area_grade["Progreso_limitado"] + area_grade["Emergente"]
        area_grade["Grado_etiqueta"] = area_grade["Grado_num"].astype(int).astype(str) + "°"
        ag = (
            area_grade.groupby(["Area", "Grado_etiqueta"], as_index=False)
            .apply(
                lambda g: pd.Series({
                    "Intentos": g["Intentos"].sum(),
                    "Promedio": (g["Promedio"] * g["Intentos"]).sum() / g["Intentos"].sum(),
                    "Requiere_apoyo": (g["Requiere_apoyo"] * g["Intentos"]).sum() / g["Intentos"].sum(),
                }),
                include_groups=False,
            )
        )
        p_area = ag.pivot(index="Area", columns="Grado_etiqueta", values="Requiere_apoyo")
        try:
            p_area = p_area[sorted(p_area.columns, key=lambda x: int(str(x).replace("°", "")))]
        except Exception:
            pass
        fig = px.imshow(
            p_area,
            text_auto=".0f",
            aspect="auto",
            zmin=0,
            zmax=100,
            labels={"x": "Grado", "y": "Área", "color": "% ≤50"},
            title="Área × grado | % en Progreso limitado + Emergente",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            ag.sort_values("Requiere_apoyo", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "La vista precargada usa información agregada y anonimizada. "
        "Para ver colegios de origen y cruces individuales, carga el Excel completo arriba; "
        "el archivo original no se publica en este repositorio público."
    )
    st.stop()

try:
    raw = read_excel(uploaded.getvalue())
    data = process_data(raw)
except Exception as exc:
    st.error(f"No fue posible procesar el archivo: {exc}")
    st.stop()

# ------------------------- Sidebar -------------------------
st.sidebar.header("Filtros")

sedes = multiselect_filter("Sede", data, "Sede", "f_sede")
grados = multiselect_filter("Grado", data, "Grado", "f_grado")
areas = multiselect_filter("Área", data, "Area", "f_area")

test_source = data.copy()
if areas:
    test_source = test_source[test_source["Area"].astype(str).isin(areas)]
pruebas = multiselect_filter("Prueba", test_source, "QuizName", "f_prueba")

has_course = (
    "Curso" in data.columns
    and data["Curso"].notna().any()
    and (data["Curso"].astype(str).str.strip() != "").any()
)
cursos = (
    multiselect_filter("Curso / grupo", data, "Curso", "f_curso")
    if has_course else []
)

filtered = apply_filters(data, sedes, grados, areas, pruebas, cursos)

if not has_course:
    st.sidebar.caption(
        "ℹ️ Esta exportación no contiene una columna de curso/grupo. "
        "El filtro aparecerá automáticamente cuando exista."
    )

if filtered.empty:
    st.warning("Los filtros seleccionados no dejan registros para analizar.")
    st.stop()

# ------------------------- KPIs -------------------------
students = filtered["IdentiEstudiante"].nunique()
attempts = filtered["AttemptId"].nunique()
avg = filtered["Porcentaje_acierto"].mean()
support = filtered["Necesita_apoyo"].mean() * 100
coverage = filtered["Cobertura_respuesta"].mean()
missing_items = filtered["No_respondidos"].sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Estudiantes", f"{students:,}".replace(",", "."))
c2.metric("Intentos", f"{attempts:,}".replace(",", "."))
c3.metric("Promedio", pct(avg))
c4.metric("En ≤50%", pct(support))
c5.metric("Cobertura", pct(coverage))
c6.metric("Ítems no observados", f"{int(missing_items):,}".replace(",", "."))

tabs = st.tabs(
    [
        "Panorama",
        "Pruebas",
        "Sedes y grados",
        "Colegios de origen",
        "Estudiantes",
        "Calidad de respuesta",
    ]
)

# ------------------------- Panorama -------------------------
with tabs[0]:
    left, right = st.columns([1.15, 1])

    with left:
        level_counts = (
            filtered["Nivel_desempeno"]
            .value_counts()
            .reindex(LEVEL_ORDER, fill_value=0)
            .rename_axis("Nivel")
            .reset_index(name="Intentos")
        )
        fig = px.bar(
            level_counts,
            x="Nivel",
            y="Intentos",
            color="Nivel",
            category_orders={"Nivel": LEVEL_ORDER},
            color_discrete_map=LEVEL_COLORS,
            text="Intentos",
            title="Distribución de niveles de desempeño",
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Intentos")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        area_summary = aggregate_group(filtered, ["Area"]).sort_values("Promedio")
        fig = px.bar(
            area_summary,
            x="Promedio",
            y="Area",
            orientation="h",
            text=area_summary["Promedio"].map(lambda x: f"{x:.1f}%"),
            title="Promedio por área",
            labels={"Area": "Área", "Promedio": "% de acierto"},
        )
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa general de retos")
    heat = aggregate_group(filtered, ["Sede", "Grado_num", "Grado"])
    if not heat.empty:
        heat["Grado_etiqueta"] = heat["Grado_num"].fillna(99).astype(int).astype(str) + "°"
        pivot = heat.pivot_table(
            index="Sede",
            columns="Grado_etiqueta",
            values="Requiere_apoyo",
            aggfunc="mean",
        )
        try:
            cols = sorted(pivot.columns, key=lambda x: int(str(x).replace("°", "")))
            pivot = pivot[cols]
        except Exception:
            pass
        fig = px.imshow(
            pivot,
            text_auto=".0f",
            aspect="auto",
            zmin=0,
            zmax=100,
            labels=dict(
                x="Grado",
                y="Sede",
                color="% en Progreso limitado + Emergente",
            ),
            title="% de intentos en niveles que requieren mayor apoyo",
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------- Pruebas -------------------------
with tabs[1]:
    summary = aggregate_group(filtered, ["Area", "QuizName", "Grado_num", "Grado"])
    summary = summary.sort_values(["Grado_num", "Area", "Promedio"])

    fig = px.scatter(
        summary,
        x="Promedio",
        y="QuizName",
        size="Estudiantes",
        color="Requiere_apoyo",
        hover_data=[
            "Grado",
            "Estudiantes",
            "Cobertura_promedio",
            "Progreso_limitado",
            "Emergente",
            "En_aceleracion",
            "Avanzado",
        ],
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
        labels={
            "Promedio": "% acierto",
            "QuizName": "Prueba",
            "Requiere_apoyo": "% ≤50",
        },
        title="Pruebas: desempeño promedio y concentración de estudiantes en ≤50%",
    )
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    selected_test = st.selectbox(
        "Explorar una prueba",
        sorted(filtered["QuizName"].dropna().astype(str).unique()),
        key="selected_test",
    )
    dtest = filtered[filtered["QuizName"].astype(str) == selected_test]

    level_by_site = (
        dtest.groupby(["Sede", "Nivel_desempeno"], as_index=False)
        .size()
        .rename(columns={"size": "Intentos"})
    )
    totals = level_by_site.groupby("Sede")["Intentos"].transform("sum")
    level_by_site["Porcentaje"] = level_by_site["Intentos"] / totals * 100

    fig = px.bar(
        level_by_site,
        x="Sede",
        y="Porcentaje",
        color="Nivel_desempeno",
        category_orders={"Nivel_desempeno": LEVEL_ORDER},
        color_discrete_map=LEVEL_COLORS,
        barmode="stack",
        title=f"Niveles de desempeño por sede | {selected_test}",
        labels={"Nivel_desempeno": "Nivel", "Porcentaje": "%"},
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- Sedes y grados -------------------------
with tabs[2]:
    dimensions = ["Sede", "Grado"]
    if has_course:
        dimensions.append("Curso")

    selected_dimension = st.radio(
        "Dimensión",
        dimensions,
        horizontal=True,
        key="challenge_dimension",
    )

    group = aggregate_group(filtered, [selected_dimension])
    group = group.sort_values(
        ["Requiere_apoyo", "Promedio"],
        ascending=[False, True],
    )

    fig = px.bar(
        group,
        x=selected_dimension,
        y="Requiere_apoyo",
        text=group["Requiere_apoyo"].map(lambda x: f"{x:.1f}%"),
        title=f"Concentración de retos por {selected_dimension.lower()}",
        labels={"Requiere_apoyo": "% en Progreso limitado + Emergente"},
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        selected_dimension,
        "Estudiantes",
        "Promedio",
        "Cobertura_promedio",
        "Progreso_limitado",
        "Emergente",
        "En_aceleracion",
        "Avanzado",
        "Requiere_apoyo",
    ]
    st.dataframe(
        group[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Promedio": st.column_config.NumberColumn(format="%.1f%%"),
            "Cobertura_promedio": st.column_config.NumberColumn(format="%.1f%%"),
            "Progreso_limitado": st.column_config.NumberColumn(format="%.1f%%"),
            "Emergente": st.column_config.NumberColumn(format="%.1f%%"),
            "En_aceleracion": st.column_config.NumberColumn(format="%.1f%%"),
            "Avanzado": st.column_config.NumberColumn(format="%.1f%%"),
            "Requiere_apoyo": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

# ------------------------- Colegios de origen -------------------------
with tabs[3]:
    st.subheader("¿De qué colegios vienen y cómo están llegando?")
    st.caption(
        "Esta vista cruza procedencia con volumen, sede de llegada, grado y desempeño. "
        "Usa el mínimo de estudiantes para evitar conclusiones con muestras demasiado pequeñas."
    )

    origin = filtered[
        filtered["colegio_de_origen"].notna()
        & (filtered["colegio_de_origen"].astype(str).str.strip() != "")
    ].copy()

    if origin.empty:
        st.info("No hay información de colegio de origen en el filtro actual.")
    else:
        student_origin = origin[
            ["IdentiEstudiante", "colegio_de_origen", "Sede", "Grado"]
        ].drop_duplicates()

        m1, m2, m3 = st.columns(3)
        m1.metric("Colegios de origen", student_origin["colegio_de_origen"].nunique())
        m2.metric("Estudiantes con procedencia", student_origin["IdentiEstudiante"].nunique())
        m3.metric(
            "Cobertura de procedencia",
            f"{student_origin['IdentiEstudiante'].nunique() / filtered['IdentiEstudiante'].nunique() * 100:.1f}%"
        )

        min_students = st.slider(
            "Mínimo de estudiantes por colegio para comparar desempeño",
            min_value=2,
            max_value=20,
            value=5,
        )

        volume = (
            student_origin.groupby("colegio_de_origen", as_index=False)
            .agg(Estudiantes=("IdentiEstudiante", "nunique"))
            .sort_values("Estudiantes", ascending=False)
        )

        top_volume = volume.head(25).sort_values("Estudiantes")
        fig = px.bar(
            top_volume,
            x="Estudiantes",
            y="colegio_de_origen",
            orientation="h",
            text="Estudiantes",
            title="Principales colegios de origen por número de estudiantes",
            labels={"colegio_de_origen": "Colegio de origen"},
        )
        st.plotly_chart(fig, use_container_width=True)

        school_summary = aggregate_group(origin, ["colegio_de_origen"])
        school_summary = school_summary[
            school_summary["Estudiantes"] >= min_students
        ].copy()
        school_summary["Pct_50_o_menos"] = (
            school_summary["Progreso_limitado"] + school_summary["Emergente"]
        )

        if not school_summary.empty:
            plot_school = school_summary.sort_values(
                ["Pct_50_o_menos", "Estudiantes"],
                ascending=[False, False],
            ).head(30)

            fig = px.scatter(
                plot_school,
                x="Promedio",
                y="colegio_de_origen",
                size="Estudiantes",
                color="Pct_50_o_menos",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 100],
                hover_data=["Estudiantes", "Cobertura_promedio"],
                title="Desempeño según colegio de origen",
                labels={
                    "Promedio": "% de acierto",
                    "colegio_de_origen": "Colegio de origen",
                    "Pct_50_o_menos": "% ≤50",
                },
            )
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Tabla: procedencia y desempeño")
            st.dataframe(
                school_summary.sort_values(
                    ["Estudiantes", "Pct_50_o_menos"],
                    ascending=[False, False],
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Promedio": st.column_config.NumberColumn(format="%.1f%%"),
                    "Cobertura_promedio": st.column_config.NumberColumn(format="%.1f%%"),
                    "Progreso_limitado": st.column_config.NumberColumn(format="%.1f%%"),
                    "Emergente": st.column_config.NumberColumn(format="%.1f%%"),
                    "En_aceleracion": st.column_config.NumberColumn(format="%.1f%%"),
                    "Avanzado": st.column_config.NumberColumn(format="%.1f%%"),
                    "Pct_50_o_menos": st.column_config.NumberColumn("% ≤50", format="%.1f%%"),
                },
            )

        st.subheader("Colegio de origen × sede de llegada")
        top_schools = volume.head(20)["colegio_de_origen"]
        school_site = (
            student_origin[student_origin["colegio_de_origen"].isin(top_schools)]
            .groupby(["colegio_de_origen", "Sede"], as_index=False)
            .agg(Estudiantes=("IdentiEstudiante", "nunique"))
        )
        site_pivot = school_site.pivot(
            index="colegio_de_origen",
            columns="Sede",
            values="Estudiantes",
        ).fillna(0)
        fig = px.imshow(
            site_pivot,
            text_auto=".0f",
            aspect="auto",
            labels={"x": "Sede de llegada", "y": "Colegio de origen", "color": "Estudiantes"},
            title="¿A qué sede llegan los estudiantes de los principales colegios de origen?",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Colegio de origen × grado de llegada")
        school_grade = (
            student_origin[student_origin["colegio_de_origen"].isin(top_schools)]
            .groupby(["colegio_de_origen", "Grado"], as_index=False)
            .agg(Estudiantes=("IdentiEstudiante", "nunique"))
        )
        st.dataframe(
            school_grade.sort_values(
                ["colegio_de_origen", "Estudiantes"],
                ascending=[True, False],
            ),
            use_container_width=True,
            hide_index=True,
        )
# ------------------------- Estudiantes -------------------------
with tabs[4]:
    student_summary = (
        filtered.groupby(
            [
                "IdentiEstudiante",
                "Estudiante",
                "Sede",
                "Grado",
                "Curso",
                "AntiguedadBS",
                "colegio_de_origen",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            Pruebas=("QuizName", "nunique"),
            Promedio=("Porcentaje_acierto", "mean"),
            Cobertura=("Cobertura_respuesta", "mean"),
            Pruebas_en_50_o_menos=("Necesita_apoyo", "sum"),
        )
    )
    student_summary["Pct_pruebas_en_50_o_menos"] = (
        student_summary["Pruebas_en_50_o_menos"] / student_summary["Pruebas"] * 100
    )
    student_summary = student_summary.sort_values(
        ["Pct_pruebas_en_50_o_menos", "Promedio"],
        ascending=[False, True],
    )

    st.dataframe(
        student_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Promedio": st.column_config.NumberColumn(format="%.1f%%"),
            "Cobertura": st.column_config.NumberColumn(format="%.1f%%"),
            "Pct_pruebas_en_50_o_menos": st.column_config.NumberColumn(
                "% pruebas ≤50", format="%.1f%%"
            ),
        },
    )
    dataframe_download(
        student_summary,
        "Descargar caracterización de estudiantes",
        "caracterizacion_estudiantes.csv",
    )

    st.divider()
    st.subheader("Perfil individual")
    choices = (
        student_summary[["IdentiEstudiante", "Estudiante"]]
        .drop_duplicates()
        .sort_values("Estudiante")
    )
    option_map = {
        f"{r.Estudiante} · {r.IdentiEstudiante}": r.IdentiEstudiante
        for r in choices.itertuples()
    }
    selected_label = st.selectbox("Estudiante", option_map.keys())
    sid = option_map[selected_label]
    one = filtered[filtered["IdentiEstudiante"] == sid].sort_values("QuizName")

    fig = px.bar(
        one,
        x="QuizName",
        y="Porcentaje_acierto",
        color="Nivel_desempeno",
        category_orders={"Nivel_desempeno": LEVEL_ORDER},
        color_discrete_map=LEVEL_COLORS,
        text=one["Porcentaje_acierto"].map(lambda x: f"{x:.1f}%"),
        title="Perfil de desempeño por prueba",
        labels={"QuizName": "Prueba", "Porcentaje_acierto": "% acierto"},
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- Calidad de respuesta -------------------------
with tabs[5]:
    q1, q2 = st.columns(2)
    with q1:
        completion = (
            filtered.groupby("QuizName", as_index=False)
            .agg(
                Cobertura=("Cobertura_respuesta", "mean"),
                No_respondidos=("No_respondidos", "sum"),
                Intentos=("AttemptId", "nunique"),
            )
            .sort_values("Cobertura")
        )
        fig = px.bar(
            completion,
            x="Cobertura",
            y="QuizName",
            orientation="h",
            text=completion["Cobertura"].map(lambda x: f"{x:.1f}%"),
            title="Cobertura promedio de respuesta por prueba",
            labels={"Cobertura": "% ítems observados", "QuizName": "Prueba"},
        )
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with q2:
        fig = px.scatter(
            filtered,
            x="Cobertura_respuesta",
            y="Porcentaje_acierto",
            color="Nivel_desempeno",
            category_orders={"Nivel_desempeno": LEVEL_ORDER},
            color_discrete_map=LEVEL_COLORS,
            hover_data=["Estudiante", "QuizName", "Sede", "Grado", "Aciertos"],
            opacity=0.45,
            title="Cobertura vs. desempeño",
            labels={
                "Cobertura_respuesta": "% ítems observados",
                "Porcentaje_acierto": "% acierto sobre denominador esperado",
            },
        )
        fig.update_xaxes(range=[0, 102])
        fig.update_yaxes(range=[0, 102])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Auditoría del cálculo por intento")
    audit_cols = [
        "AttemptId",
        "IdentiEstudiante",
        "Estudiante",
        "Sede",
        "Grado",
        "QuizName",
        "Aciertos",
        "Items_observados",
        "Items_esperados",
        "No_respondidos",
        "Porcentaje_acierto",
        "Nivel_desempeno",
        "Cobertura_respuesta",
    ]
    st.dataframe(
        filtered[audit_cols].sort_values(
            ["No_respondidos", "Porcentaje_acierto"],
            ascending=[False, True],
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Porcentaje_acierto": st.column_config.NumberColumn(format="%.1f%%"),
            "Cobertura_respuesta": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    dataframe_download(
        filtered[audit_cols],
        "Descargar auditoría de intentos",
        "auditoria_intentos.csv",
    )
