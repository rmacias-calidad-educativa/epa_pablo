
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from processing import aggregate_group, build_attempt_table, normalize_area
from ui import (
    LEVEL_COLORS,
    LEVEL_ORDER,
    apply_filters,
    dataframe_download,
    multiselect_filter,
)

CORE_TESTS = [
    "Ciencias naturales",
    "Ciencias sociales",
    "Matemáticas",
    "Lenguaje",
]


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
    data = build_attempt_table(raw)
    return data[data["QuizName"].isin(CORE_TESTS)].copy()


def pct(x):
    return f"{x:.1f}%" if pd.notna(x) else "—"


def num(x, digits=1):
    return f"{x:.{digits}f}" if pd.notna(x) else "—"

def render_grade_line_chart(
    source: pd.DataFrame,
    key_prefix: str,
    aggregated: bool = False,
):
    """Trayectoria por grado del número de estudiantes con ≤50%."""
    sites = ["BAQ", "COT", "MOS", "TUN", "USAQ", "ZIPA"]

    d = source.copy()
    if "QuizName" in d.columns:
        d["Prueba"] = d["QuizName"].apply(normalize_area)
    elif "Area" in d.columns:
        d["Prueba"] = d["Area"].apply(normalize_area)
    else:
        st.info("No hay una columna de prueba disponible.")
        return

    d = d[d["Prueba"].isin(CORE_TESTS)].copy()
    if d.empty:
        st.info("No hay datos para las cuatro pruebas definidas.")
        return

    st.subheader("Trayectoria de estudiantes con ≤50% a lo largo de los grados")
    st.caption(
        "Cada línea representa una de las seis sedes/colegios. "
        "Selecciona una prueba para evitar saturación visual. "
        "El eje Y muestra la cantidad de estudiantes con 50% o menos de aciertos."
    )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        selected_test = st.selectbox(
            "Prueba",
            CORE_TESTS,
            key=f"{key_prefix}_test",
        )

    years = (
        sorted([int(y) for y in d["Año"].dropna().unique()])
        if "Año" in d.columns
        else []
    )
    with c2:
        selected_year = st.selectbox(
            "Año",
            ["Todos"] + years,
            index=len(years),
            key=f"{key_prefix}_year",
        )

    d = d[d["Prueba"] == selected_test].copy()
    if selected_year != "Todos" and "Año" in d.columns:
        d = d[d["Año"] == selected_year]

    if d.empty:
        st.info("No hay registros para la prueba y el año seleccionados.")
        return

    if aggregated:
        grouped = (
            d.groupby(["Grado_num", "Sede"], as_index=False)
            .agg(
                Estudiantes=("Estudiantes", "sum"),
                Menor_igual_50=("Debajo_igual_50", "sum"),
                Mayor_50=("Encima_50", "sum"),
            )
        )
    else:
        # Clasificación a nivel de estudiante, prueba y grado.
        student_test = (
            d.groupby(
                ["IdentiEstudiante", "Sede", "Grado_num"],
                as_index=False,
            )
            .agg(Porcentaje=("Porcentaje_acierto", "mean"))
        )
        student_test["Menor_igual_50"] = student_test["Porcentaje"] <= 50
        grouped = (
            student_test.groupby(["Grado_num", "Sede"], as_index=False)
            .agg(
                Estudiantes=("IdentiEstudiante", "nunique"),
                Menor_igual_50=("Menor_igual_50", "sum"),
            )
        )
        grouped["Mayor_50"] = (
            grouped["Estudiantes"] - grouped["Menor_igual_50"]
        )

    grouped["Pct_50_o_menos"] = np.where(
        grouped["Estudiantes"] > 0,
        grouped["Menor_igual_50"] / grouped["Estudiantes"] * 100,
        np.nan,
    )
    grouped["Grado"] = (
        grouped["Grado_num"].astype(int).astype(str) + "°"
    )

    total_under = int(grouped["Menor_igual_50"].sum())
    total_above = int(grouped["Mayor_50"].sum())
    total_students = total_under + total_above

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estudiantes ≤50%", f"{total_under:,}".replace(",", "."))
    k2.metric("Estudiantes >50%", f"{total_above:,}".replace(",", "."))
    k3.metric(
        "% ≤50%",
        f"{total_under / total_students * 100:.1f}%"
        if total_students else "—",
    )
    k4.metric("Prueba", selected_test)

    fig = px.line(
        grouped.sort_values(["Sede", "Grado_num"]),
        x="Grado_num",
        y="Menor_igual_50",
        color="Sede",
        markers=True,
        category_orders={"Sede": sites},
        hover_data={
            "Grado_num": False,
            "Grado": True,
            "Estudiantes": True,
            "Menor_igual_50": True,
            "Mayor_50": True,
            "Pct_50_o_menos": ":.1f",
        },
        labels={
            "Grado_num": "Grado",
            "Menor_igual_50": "Estudiantes ≤50%",
            "Sede": "Sede/colegio",
            "Mayor_50": "Estudiantes >50%",
            "Pct_50_o_menos": "% ≤50%",
        },
        title=f"Estudiantes con ≤50% por grado y sede | {selected_test}",
    )
    grades = sorted(
        [int(g) for g in grouped["Grado_num"].dropna().unique()]
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=grades,
        ticktext=[f"{g}°" for g in grades],
    )
    fig.update_yaxes(rangemode="tozero", dtick=1)
    fig.update_layout(
        height=620,
        legend_title_text="Sede/colegio",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "El conteo es de estudiantes, no de intentos. "
        "En el tooltip puedes ver también cuántos estudiantes quedaron >50% "
        "y el porcentaje ≤50% para cada sede y grado."
    )

    if selected_year == 2025 and total_students < 30:
        st.info(
            "La base actual tiene muy pocos registros de 2025. "
            "Interpreta esa trayectoria únicamente de forma descriptiva."
        )


st.title("📊 Estado de llegada de los estudiantes")
st.caption(
    "Dashboard diagnóstico para responder tres preguntas: cómo llegan los estudiantes, "
    "dónde se concentran los retos y de qué colegios provienen quienes requieren mayor apoyo."
)

with st.expander("Reglas de cálculo", expanded=False):
    st.markdown(
        """
- **Progreso limitado:** 0% a 25% (incluye 25%).
- **Emergente:** >25% a 50%.
- **En aceleración:** >50% a 75%.
- **Avanzado:** >75% a 100%.
- **Pruebas visibles:** Ciencias naturales, Ciencias sociales, Matemáticas y Lenguaje.
- **Denominador:** 20 ítems en las cuatro pruebas.
- El grado se analiza como una dimensión independiente y no forma parte del nombre de la prueba.
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
    summary_data["Area"] = summary_data["Area"].apply(normalize_area)
    summary_data = summary_data[summary_data["Area"].isin(CORE_TESTS)].copy()
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

    temporal_path = Path("data/default_temporal_summary.csv")
    if temporal_path.exists():
        temporal_data = pd.read_csv(temporal_path)
        temporal_data["QuizName"] = temporal_data["QuizName"].apply(normalize_area)
        temporal_data = temporal_data[
            temporal_data["QuizName"].isin(CORE_TESTS)
        ].copy()
        render_grade_line_chart(
            temporal_data,
            key_prefix="default_grade_line",
            aggregated=True,
        )
        st.divider()

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
available_tests = [
    test for test in CORE_TESTS
    if test in test_source["QuizName"].dropna().astype(str).unique()
]
pruebas = st.sidebar.multiselect(
    "Prueba",
    available_tests,
    key="f_prueba",
)

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

st.divider()
render_grade_line_chart(
    filtered,
    key_prefix="full_grade_line",
    aggregated=False,
)
st.divider()

tabs = st.tabs(
    [
        "Panorama",
        "Pruebas",
        "Dónde están los retos",
        "Origen × nivel",
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
    st.subheader("De qué colegios vienen y en qué nivel llegan")
    st.caption(
        "Lee esta sección de arriba hacia abajo: selecciona una sede, identifica los colegios "
        "que más estudiantes aportan y observa cómo se distribuyen esos estudiantes entre "
        "Progreso limitado, Emergente, En aceleración y Avanzado."
    )

    origin = filtered[
        filtered["colegio_de_origen"].notna()
        & (filtered["colegio_de_origen"].astype(str).str.strip() != "")
    ].copy()

    if origin.empty:
        st.info("No hay información de colegio de origen en el filtro actual.")
    else:
        sedes_origin = sorted(origin["Sede"].dropna().astype(str).unique())
        selected_site_origin = st.selectbox(
            "Sede de llegada",
            ["Todas"] + sedes_origin,
            key="origin_site_selector",
        )

        origin_site = origin.copy()
        if selected_site_origin != "Todas":
            origin_site = origin_site[
                origin_site["Sede"].astype(str) == selected_site_origin
            ]

        # Una fila por estudiante para volumen y procedencia.
        student_origin = origin_site[
            ["IdentiEstudiante", "colegio_de_origen", "Sede", "Grado"]
        ].drop_duplicates()

        # Nivel global del estudiante dentro del filtro actual.
        student_level = (
            origin_site.groupby(
                ["IdentiEstudiante", "colegio_de_origen", "Sede", "Grado"],
                as_index=False,
            )
            .agg(
                Promedio_estudiante=("Porcentaje_acierto", "mean"),
                Pruebas=("QuizName", "nunique"),
            )
        )
        student_level["Nivel_estudiante"] = student_level["Promedio_estudiante"].apply(
            lambda x: (
                "Progreso limitado" if x <= 25
                else "Emergente" if x <= 50
                else "En aceleración" if x <= 75
                else "Avanzado"
            )
        )

        total_students_origin = student_origin["IdentiEstudiante"].nunique()
        total_schools_origin = student_origin["colegio_de_origen"].nunique()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sede", selected_site_origin)
        m2.metric("Estudiantes", f"{total_students_origin:,}".replace(",", "."))
        m3.metric("Colegios de origen", f"{total_schools_origin:,}".replace(",", "."))
        m4.metric(
            "Cobertura de procedencia",
            f"{total_students_origin / origin_site['IdentiEstudiante'].nunique() * 100:.1f}%"
            if origin_site["IdentiEstudiante"].nunique() else "—",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            min_students = st.slider(
                "Mínimo de estudiantes por colegio",
                min_value=2,
                max_value=20,
                value=5,
                key="origin_min_students",
            )
        with c2:
            top_n = st.slider(
                "Número máximo de colegios a mostrar",
                min_value=5,
                max_value=30,
                value=15,
                key="origin_top_n",
            )

        school_volume = (
            student_origin.groupby("colegio_de_origen", as_index=False)
            .agg(Estudiantes=("IdentiEstudiante", "nunique"))
        )
        eligible_schools = school_volume[
            school_volume["Estudiantes"] >= min_students
        ].sort_values("Estudiantes", ascending=False)

        top_schools = eligible_schools.head(top_n)["colegio_de_origen"].tolist()

        if not top_schools:
            st.warning(
                "No hay colegios con el mínimo de estudiantes seleccionado. "
                "Reduce el umbral para ampliar la comparación."
            )
        else:
            level_counts = (
                student_level[
                    student_level["colegio_de_origen"].isin(top_schools)
                ]
                .groupby(["colegio_de_origen", "Nivel_estudiante"], as_index=False)
                .agg(Estudiantes=("IdentiEstudiante", "nunique"))
            )

            totals_school = (
                level_counts.groupby("colegio_de_origen")["Estudiantes"]
                .transform("sum")
            )
            level_counts["Porcentaje"] = (
                level_counts["Estudiantes"] / totals_school * 100
            )

            school_order = (
                student_level[
                    student_level["colegio_de_origen"].isin(top_schools)
                ]
                .groupby("colegio_de_origen", as_index=False)
                .agg(
                    Estudiantes=("IdentiEstudiante", "nunique"),
                    Promedio=("Promedio_estudiante", "mean"),
                )
                .sort_values(["Estudiantes", "Promedio"], ascending=[True, True])
            )["colegio_de_origen"].tolist()

            st.subheader("1. Composición por nivel de desempeño")
            st.caption(
                "Cada barra representa el 100% de los estudiantes que llegan desde ese colegio "
                "a la sede seleccionada. Esto permite comparar composición, no solo promedios."
            )

            fig = px.bar(
                level_counts,
                x="Porcentaje",
                y="colegio_de_origen",
                color="Nivel_estudiante",
                orientation="h",
                barmode="stack",
                category_orders={
                    "Nivel_estudiante": LEVEL_ORDER,
                    "colegio_de_origen": school_order,
                },
                color_discrete_map=LEVEL_COLORS,
                text=level_counts["Porcentaje"].map(
                    lambda x: f"{x:.0f}%" if x >= 7 else ""
                ),
                hover_data={"Estudiantes": True, "Porcentaje": ":.1f"},
                title=(
                    "Distribución de estudiantes por nivel y colegio de origen"
                    + (
                        f" | Sede {selected_site_origin}"
                        if selected_site_origin != "Todas"
                        else ""
                    )
                ),
                labels={
                    "colegio_de_origen": "Colegio de origen",
                    "Porcentaje": "% de estudiantes",
                    "Nivel_estudiante": "Nivel",
                },
            )
            fig.update_xaxes(range=[0, 100])
            fig.update_layout(
                legend_title_text="Nivel de desempeño",
                height=max(520, 34 * len(school_order) + 180),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("2. Volumen y desempeño por colegio")
            school_perf = (
                student_level[
                    student_level["colegio_de_origen"].isin(top_schools)
                ]
                .groupby("colegio_de_origen", as_index=False)
                .agg(
                    Estudiantes=("IdentiEstudiante", "nunique"),
                    Promedio=("Promedio_estudiante", "mean"),
                )
            )

            support_school = (
                student_level[
                    student_level["colegio_de_origen"].isin(top_schools)
                ]
                .assign(
                    Requiere_apoyo=lambda d: d["Nivel_estudiante"].isin(
                        ["Progreso limitado", "Emergente"]
                    )
                )
                .groupby("colegio_de_origen", as_index=False)
                .agg(Pct_50_o_menos=("Requiere_apoyo", "mean"))
            )
            support_school["Pct_50_o_menos"] *= 100
            school_perf = school_perf.merge(
                support_school,
                on="colegio_de_origen",
                how="left",
            )

            left, right = st.columns([1.05, 1])

            with left:
                vol_plot = school_perf.sort_values("Estudiantes")
                fig = px.bar(
                    vol_plot,
                    x="Estudiantes",
                    y="colegio_de_origen",
                    orientation="h",
                    text="Estudiantes",
                    title="Número de estudiantes por colegio de origen",
                    labels={"colegio_de_origen": "Colegio de origen"},
                )
                fig.update_layout(
                    height=max(500, 32 * len(vol_plot) + 150)
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                risk_plot = school_perf.sort_values("Pct_50_o_menos")
                fig = px.bar(
                    risk_plot,
                    x="Pct_50_o_menos",
                    y="colegio_de_origen",
                    orientation="h",
                    text=risk_plot["Pct_50_o_menos"].map(lambda x: f"{x:.0f}%"),
                    title="% de estudiantes en Progreso limitado + Emergente",
                    labels={
                        "colegio_de_origen": "Colegio de origen",
                        "Pct_50_o_menos": "% ≤50",
                    },
                )
                fig.update_xaxes(range=[0, 100])
                fig.update_layout(
                    height=max(500, 32 * len(risk_plot) + 150)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("3. Colegio de origen × grado de llegada")
            school_grade = (
                student_origin[
                    student_origin["colegio_de_origen"].isin(top_schools)
                ]
                .groupby(["colegio_de_origen", "Grado"], as_index=False)
                .agg(Estudiantes=("IdentiEstudiante", "nunique"))
            )
            grade_pivot = school_grade.pivot(
                index="colegio_de_origen",
                columns="Grado",
                values="Estudiantes",
            ).fillna(0)

            fig = px.imshow(
                grade_pivot,
                text_auto=".0f",
                aspect="auto",
                labels={
                    "x": "Grado de llegada",
                    "y": "Colegio de origen",
                    "color": "Estudiantes",
                },
                title="Cantidad de estudiantes según colegio de origen y grado",
            )
            fig.update_layout(
                height=max(500, 32 * len(grade_pivot.index) + 160)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("4. Vista jerárquica: sede → colegio → nivel")
            hierarchy = (
                student_level[
                    student_level["colegio_de_origen"].isin(top_schools)
                ]
                .groupby(
                    ["Sede", "colegio_de_origen", "Nivel_estudiante"],
                    as_index=False,
                )
                .agg(Estudiantes=("IdentiEstudiante", "nunique"))
            )

            fig = px.treemap(
                hierarchy,
                path=["Sede", "colegio_de_origen", "Nivel_estudiante"],
                values="Estudiantes",
                color="Nivel_estudiante",
                color_discrete_map=LEVEL_COLORS,
                title="Estructura de procedencia y nivel de desempeño",
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("5. Tabla de lectura ejecutiva")
            table_school = school_perf.sort_values(
                ["Pct_50_o_menos", "Estudiantes"],
                ascending=[False, False],
            ).copy()
            table_school["Lectura"] = np.select(
                [
                    table_school["Pct_50_o_menos"] >= 60,
                    table_school["Pct_50_o_menos"] >= 40,
                ],
                [
                    "Alta concentración en ≤50%",
                    "Concentración intermedia en ≤50%",
                ],
                default="Menor concentración en ≤50%",
            )

            st.dataframe(
                table_school,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Promedio": st.column_config.NumberColumn(format="%.1f%%"),
                    "Pct_50_o_menos": st.column_config.NumberColumn(
                        "% ≤50",
                        format="%.1f%%",
                    ),
                },
            )

            dataframe_download(
                table_school,
                "Descargar resumen por colegio de origen",
                "resumen_colegios_origen.csv",
            )

        st.divider()
        st.subheader("6. ¿Hay patrones asociados al colegio de origen?")
        st.caption(
            "Esta lectura busca recurrencia. Compara, por área, el % de estudiantes ≤50% "
            "de cada colegio contra el promedio de la red en el mismo filtro."
        )

        student_area = (
            origin_site.groupby(
                ["IdentiEstudiante", "colegio_de_origen", "Area"],
                as_index=False,
            )
            .agg(Promedio_area=("Porcentaje_acierto", "mean"))
        )
        student_area["Bajo_50"] = student_area["Promedio_area"] <= 50

        network_area = (
            student_area.groupby("Area", as_index=False)
            .agg(Pct_red_50=("Bajo_50", "mean"))
        )
        network_area["Pct_red_50"] *= 100

        school_area = (
            student_area[
                student_area["colegio_de_origen"].isin(top_schools)
            ]
            .groupby(["colegio_de_origen", "Area"], as_index=False)
            .agg(
                Estudiantes=("IdentiEstudiante", "nunique"),
                Pct_50_o_menos=("Bajo_50", "mean"),
            )
        )
        school_area["Pct_50_o_menos"] *= 100
        school_area = school_area.merge(network_area, on="Area", how="left")
        school_area["Diferencia_vs_red"] = (
            school_area["Pct_50_o_menos"] - school_area["Pct_red_50"]
        )

        school_area_valid = school_area[
            school_area["Estudiantes"] >= 3
        ].copy()

        if not school_area_valid.empty:
            delta_pivot = school_area_valid.pivot(
                index="colegio_de_origen",
                columns="Area",
                values="Diferencia_vs_red",
            )
            raw_pivot = school_area_valid.pivot(
                index="colegio_de_origen",
                columns="Area",
                values="Pct_50_o_menos",
            )

            text_pattern = []
            for school in delta_pivot.index:
                row = []
                for area in delta_pivot.columns:
                    raw_value = raw_pivot.loc[school, area]
                    row.append(
                        "—" if pd.isna(raw_value) else f"{raw_value:.0f}% ≤50"
                    )
                text_pattern.append(row)

            max_abs = np.nanmax(np.abs(delta_pivot.values))
            max_abs = max(20, min(60, max_abs if np.isfinite(max_abs) else 20))

            fig = go.Figure(
                data=go.Heatmap(
                    z=delta_pivot.values,
                    x=delta_pivot.columns,
                    y=delta_pivot.index,
                    zmid=0,
                    zmin=-max_abs,
                    zmax=max_abs,
                    colorscale="RdBu_r",
                    text=text_pattern,
                    texttemplate="%{text}",
                    hovertemplate=(
                        "Colegio: %{y}<br>Área: %{x}<br>"
                        "Diferencia vs red: %{z:+.1f} pp<br>%{text}<extra></extra>"
                    ),
                    colorbar=dict(title="Δ pp vs red"),
                )
            )
            fig.update_layout(
                title="Patrón por colegio y área: diferencia en %≤50 frente a la red",
                xaxis_title="Área",
                yaxis_title="",
                height=max(520, 34 * len(delta_pivot.index) + 180),
            )
            st.plotly_chart(fig, use_container_width=True)

            repeat_attempts = (
                origin_site[
                    origin_site["colegio_de_origen"].isin(top_schools)
                ]
                .assign(Bajo_50=lambda d: d["Porcentaje_acierto"] <= 50)
                .groupby(
                    ["IdentiEstudiante", "colegio_de_origen"],
                    as_index=False,
                )
                .agg(
                    Pruebas=("QuizName", "nunique"),
                    Pruebas_50_o_menos=("Bajo_50", "sum"),
                )
            )
            repeat_attempts["Recurrente_2mas"] = (
                repeat_attempts["Pruebas_50_o_menos"] >= 2
            )
            repeat_school = (
                repeat_attempts.groupby("colegio_de_origen", as_index=False)
                .agg(
                    Estudiantes=("IdentiEstudiante", "nunique"),
                    Pct_estudiantes_2mas=("Recurrente_2mas", "mean"),
                )
            )
            repeat_school["Pct_estudiantes_2mas"] *= 100

            recurrent_areas = (
                school_area_valid.assign(
                    Area_sobre_red_10pp=lambda d: d["Diferencia_vs_red"] >= 10
                )
                .groupby("colegio_de_origen", as_index=False)
                .agg(
                    Areas_evaluadas=("Area", "nunique"),
                    Areas_sobre_red_10pp=("Area_sobre_red_10pp", "sum"),
                    Mayor_diferencia_pp=("Diferencia_vs_red", "max"),
                )
            )

            pattern_summary = repeat_school.merge(
                recurrent_areas,
                on="colegio_de_origen",
                how="left",
            )
            pattern_summary["Patron_descriptivo"] = np.select(
                [
                    pattern_summary["Areas_sobre_red_10pp"] >= 2,
                    pattern_summary["Areas_sobre_red_10pp"] == 1,
                ],
                [
                    "Recurrente en varias áreas",
                    "Concentrado en un área",
                ],
                default="Sin patrón recurrente claro",
            )

            st.dataframe(
                pattern_summary.sort_values(
                    ["Areas_sobre_red_10pp", "Pct_estudiantes_2mas"],
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pct_estudiantes_2mas": st.column_config.NumberColumn(
                        "% estudiantes con 2+ pruebas ≤50",
                        format="%.1f%%",
                    ),
                    "Mayor_diferencia_pp": st.column_config.NumberColumn(
                        "Mayor diferencia vs red",
                        format="%+.1f pp",
                    ),
                },
            )

            # Asociación descriptiva global colegio de origen ↔ nivel.
            contingency = pd.crosstab(
                student_level[
                    student_level["colegio_de_origen"].isin(top_schools)
                ]["colegio_de_origen"],
                student_level[
                    student_level["colegio_de_origen"].isin(top_schools)
                ]["Nivel_estudiante"],
            )
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                observed = contingency.to_numpy(dtype=float)
                n = observed.sum()
                expected = (
                    observed.sum(axis=1, keepdims=True)
                    @ observed.sum(axis=0, keepdims=True)
                    / n
                )
                valid = expected > 0
                chi2 = np.sum(
                    ((observed - expected) ** 2 / np.where(valid, expected, 1))[valid]
                )
                denom = min(observed.shape[0] - 1, observed.shape[1] - 1)
                cramers_v = np.sqrt(chi2 / (n * denom)) if denom > 0 else np.nan
                st.metric(
                    "Asociación descriptiva colegio de origen ↔ nivel (V de Cramér)",
                    f"{cramers_v:.3f}" if pd.notna(cramers_v) else "—",
                )
                st.caption(
                    "V de Cramér va de 0 a 1. Describe cuánto se apartan las distribuciones "
                    "de nivel entre colegios; no prueba causalidad y puede variar con muestras pequeñas."
                )
        else:
            st.info(
                "No hay suficientes estudiantes por colegio y área para evaluar patrones "
                "con el umbral actual."
            )

        st.divider()
        st.caption(
            "El nivel del estudiante en esta vista se calcula con el promedio de sus pruebas "
            "dentro de los filtros activos. Los patrones son descriptivos y deben leerse junto "
            "con el tamaño de muestra del colegio."
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
