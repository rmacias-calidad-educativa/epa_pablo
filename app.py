
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from processing import build_attempt_table, normalize_area


YEAR = 2026
TESTS = [
    "Ciencias naturales",
    "Ciencias sociales",
    "Matemáticas",
    "Lenguaje",
    "Inglés",
]
SITES = ["BAQ", "COT", "MOS", "TUN", "USAQ", "ZIPA"]
LEVEL_ORDER = [
    "Progreso limitado",
    "Emergente",
    "En aceleración",
    "Avanzado",
]
LEVEL_COLORS = {
    "Progreso limitado": "#C93C3C",
    "Emergente": "#E58B2A",
    "En aceleración": "#D8B531",
    "Avanzado": "#2E8B57",
}


st.set_page_config(
    page_title="Estado de llegada 2026",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1550px;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.16);
        padding: 10px 12px;
        border-radius: 12px;
    }
    .muted {
        color: #777;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def process_raw(raw: pd.DataFrame) -> pd.DataFrame:
    data = build_attempt_table(raw)
    data["Prueba"] = data["QuizName"].apply(normalize_area)
    data = data[
        (data["Año"] == YEAR)
        & (data["Prueba"].isin(TESTS))
    ].copy()
    return data


@st.cache_data(show_spinner=False)
def read_default_summary() -> pd.DataFrame:
    path = Path("data/default_temporal_summary.csv")
    if not path.exists():
        return pd.DataFrame()

    d = pd.read_csv(path)
    d["Prueba"] = d["QuizName"].apply(normalize_area)
    if "Año" in d.columns:
        d = d[d["Año"] == YEAR]
    return d[d["Prueba"].isin(TESTS)].copy()


def grade_label(value) -> str:
    try:
        return f"{int(float(value))}°"
    except Exception:
        return str(value)


def clean_line_layout(fig, grades):
    fig.update_xaxes(
        title=None,
        tickmode="array",
        tickvals=grades,
        ticktext=[grade_label(g) for g in grades],
        showgrid=False,
        zeroline=False,
        ticks="",
    )
    fig.update_yaxes(
        title=None,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        rangemode="tozero",
    )
    fig.update_layout(
        height=610,
        hovermode="x unified",
        legend_title_text="Sede",
        margin=dict(l=18, r=18, t=70, b=25),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(
        textposition="top center",
        cliponaxis=False,
        line=dict(width=2.5),
        marker=dict(size=8),
    )
    return fig


def build_student_test_grade(data: pd.DataFrame) -> pd.DataFrame:
    student = (
        data.groupby(
            ["IdentiEstudiante", "Sede", "Grado_num"],
            as_index=False,
        )
        .agg(
            Porcentaje=("Porcentaje_acierto", "mean"),
            colegio_de_origen=("colegio_de_origen", "first"),
        )
    )
    student["Menor_igual_50"] = student["Porcentaje"] <= 50
    student["Nivel"] = np.select(
        [
            student["Porcentaje"] <= 25,
            student["Porcentaje"] <= 50,
            student["Porcentaje"] <= 75,
        ],
        [
            "Progreso limitado",
            "Emergente",
            "En aceleración",
        ],
        default="Avanzado",
    )
    return student


def render_main_line(d: pd.DataFrame, aggregated: bool):
    if aggregated:
        grouped = (
            d.groupby(["Sede", "Grado_num"], as_index=False)
            .agg(
                Estudiantes=("Estudiantes", "sum"),
                Menor_igual_50=("Debajo_igual_50", "sum"),
                Mayor_50=("Encima_50", "sum"),
            )
        )
    else:
        student = build_student_test_grade(d)
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

    if grouped.empty:
        st.info("No hay datos para los filtros seleccionados.")
        return

    grouped["Pct_50_o_menos"] = np.where(
        grouped["Estudiantes"] > 0,
        grouped["Menor_igual_50"] / grouped["Estudiantes"] * 100,
        np.nan,
    )
    grouped["Grado"] = grouped["Grado_num"].apply(grade_label)

    total_low = int(grouped["Menor_igual_50"].sum())
    total_high = int(grouped["Mayor_50"].sum())
    total = total_low + total_high

    m1, m2, m3 = st.columns(3)
    m1.metric("Estudiantes ≤50%", f"{total_low:,}".replace(",", "."))
    m2.metric("Estudiantes >50%", f"{total_high:,}".replace(",", "."))
    m3.metric(
        "% ≤50%",
        f"{(total_low / total * 100):.1f}%" if total else "—",
    )

    grades = sorted(
        [int(g) for g in grouped["Grado_num"].dropna().unique()]
    )

    fig = px.line(
        grouped.sort_values(["Sede", "Grado_num"]),
        x="Grado_num",
        y="Menor_igual_50",
        color="Sede",
        markers=True,
        text="Menor_igual_50",
        category_orders={"Sede": SITES},
        hover_data={
            "Grado_num": False,
            "Grado": True,
            "Estudiantes": True,
            "Menor_igual_50": True,
            "Mayor_50": True,
            "Pct_50_o_menos": ":.1f",
        },
        labels={
            "Sede": "Sede",
            "Menor_igual_50": "≤50%",
            "Mayor_50": ">50%",
            "Pct_50_o_menos": "% ≤50%",
        },
        title="Cantidad de estudiantes con ≤50% de aciertos por grado",
    )
    fig = clean_line_layout(fig, grades)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Las cifras sobre cada punto son estudiantes con ≤50% de aciertos. "
        "El detalle emergente muestra también el total y los estudiantes >50%."
    )


def render_level_distribution(d: pd.DataFrame):
    student = build_student_test_grade(d)
    if student.empty:
        return

    levels = (
        student.groupby(["Grado_num", "Nivel"], as_index=False)
        .agg(Estudiantes=("IdentiEstudiante", "nunique"))
    )
    totals = levels.groupby("Grado_num")["Estudiantes"].transform("sum")
    levels["Porcentaje"] = levels["Estudiantes"] / totals * 100
    levels["Grado"] = levels["Grado_num"].apply(grade_label)

    fig = px.bar(
        levels,
        x="Grado",
        y="Porcentaje",
        color="Nivel",
        text="Estudiantes",
        barmode="stack",
        category_orders={"Nivel": LEVEL_ORDER},
        color_discrete_map=LEVEL_COLORS,
        labels={"Grado": "", "Porcentaje": "", "Nivel": "Nivel"},
        title="Distribución de niveles de desempeño por grado",
    )
    fig.update_yaxes(
        range=[0, 100],
        showticklabels=False,
        showgrid=False,
        title=None,
        ticks="",
    )
    fig.update_xaxes(
        title=None,
        showgrid=False,
        ticks="",
    )
    fig.update_layout(
        height=480,
        margin=dict(l=18, r=18, t=65, b=25),
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Nivel",
    )
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)


def render_origin_patterns(d: pd.DataFrame):
    if (
        "colegio_de_origen" not in d.columns
        or d["colegio_de_origen"].dropna().empty
    ):
        st.info(
            "Para ver patrones por colegio de origen, carga el Excel completo."
        )
        return

    origin = d[
        d["colegio_de_origen"].notna()
        & (d["colegio_de_origen"].astype(str).str.strip() != "")
    ].copy()

    if origin.empty:
        st.info("No hay información de colegio de origen en este filtro.")
        return

    st.subheader("Patrones por colegio de origen")
    st.caption(
        "Aquí se busca recurrencia dentro de la misma prueba: "
        "si estudiantes de un mismo colegio tienden a concentrarse en ≤50% "
        "a través de varios grados."
    )

    student = (
        origin.groupby(
            [
                "IdentiEstudiante",
                "colegio_de_origen",
                "Sede",
                "Grado_num",
            ],
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

    c1, c2 = st.columns(2)
    with c1:
        min_n = st.slider(
            "Mínimo de estudiantes por colegio de origen",
            2,
            20,
            5,
            key=f"min_origin_{st.session_state.get('_active_test', 'x')}",
        )
    with c2:
        top_n = st.slider(
            "Número de colegios a comparar",
            3,
            15,
            8,
            key=f"top_origin_{st.session_state.get('_active_test', 'x')}",
        )

    eligible = school_volume[
        school_volume["Estudiantes"] >= min_n
    ].head(top_n)

    if eligible.empty:
        st.info("No hay colegios de origen con el tamaño mínimo seleccionado.")
        return

    schools = eligible["colegio_de_origen"].tolist()
    sd = student[student["colegio_de_origen"].isin(schools)].copy()

    trend = (
        sd.groupby(["colegio_de_origen", "Grado_num"], as_index=False)
        .agg(
            Estudiantes=("IdentiEstudiante", "nunique"),
            Menor_igual_50=("Bajo_50", "sum"),
        )
    )
    trend["Pct_50"] = np.where(
        trend["Estudiantes"] > 0,
        trend["Menor_igual_50"] / trend["Estudiantes"] * 100,
        np.nan,
    )

    grades = sorted(
        [int(g) for g in trend["Grado_num"].dropna().unique()]
    )
    fig = px.line(
        trend.sort_values(["colegio_de_origen", "Grado_num"]),
        x="Grado_num",
        y="Pct_50",
        color="colegio_de_origen",
        markers=True,
        text=trend["Pct_50"].map(lambda x: f"{x:.0f}%"),
        hover_data={
            "Grado_num": False,
            "Estudiantes": True,
            "Menor_igual_50": True,
            "Pct_50": ":.1f",
        },
        labels={
            "colegio_de_origen": "Colegio de origen",
            "Pct_50": "% ≤50",
        },
        title="% de estudiantes ≤50% por grado y colegio de origen",
    )
    fig.update_xaxes(
        title=None,
        tickmode="array",
        tickvals=grades,
        ticktext=[grade_label(g) for g in grades],
        showgrid=False,
        ticks="",
    )
    fig.update_yaxes(
        title=None,
        range=[0, 100],
        showticklabels=False,
        showgrid=False,
        ticks="",
        zeroline=False,
    )
    fig.update_layout(
        height=560,
        margin=dict(l=18, r=18, t=70, b=25),
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Colegio de origen",
    )
    fig.update_traces(
        textposition="top center",
        cliponaxis=False,
        line=dict(width=2.2),
        marker=dict(size=7),
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = (
        sd.groupby("colegio_de_origen", as_index=False)
        .agg(
            Estudiantes=("IdentiEstudiante", "nunique"),
            Grados_observados=("Grado_num", "nunique"),
            Pct_50=("Bajo_50", "mean"),
        )
    )
    summary["Pct_50"] *= 100

    recurrent = (
        trend.assign(Grado_concentrado=lambda x: x["Pct_50"] >= 50)
        .groupby("colegio_de_origen", as_index=False)
        .agg(
            Grados_con_50pct_o_mas_en_bajo50=(
                "Grado_concentrado",
                "sum",
            )
        )
    )
    summary = summary.merge(recurrent, on="colegio_de_origen", how="left")
    summary["Patron"] = np.select(
        [
            summary["Grados_con_50pct_o_mas_en_bajo50"] >= 2,
            summary["Grados_con_50pct_o_mas_en_bajo50"] == 1,
        ],
        [
            "Recurrente en varios grados",
            "Concentrado en un grado",
        ],
        default="Sin patrón recurrente claro",
    )

    st.dataframe(
        summary.sort_values(
            ["Grados_con_50pct_o_mas_en_bajo50", "Pct_50"],
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pct_50": st.column_config.NumberColumn(
                "% estudiantes ≤50",
                format="%.1f%%",
            ),
        },
    )


def render_test_page(
    data: pd.DataFrame,
    test_name: str,
    aggregated: bool,
):
    st.session_state["_active_test"] = test_name

    st.markdown(f"## {test_name}")
    st.caption(
        f"Resultados de {test_name} · Año fijo: {YEAR}"
    )

    d = data[data["Prueba"] == test_name].copy()
    if d.empty:
        st.info(f"No hay datos de {test_name} para {YEAR}.")
        return

    filter_cols = st.columns([1.2, 1.5])
    with filter_cols[0]:
        available_sites = [
            s for s in SITES if s in d["Sede"].dropna().astype(str).unique()
        ]
        selected_sites = st.multiselect(
            "Sedes/colegios",
            available_sites,
            default=available_sites,
            key=f"sites_{test_name}",
        )

    with filter_cols[1]:
        grades = sorted(
            [int(g) for g in d["Grado_num"].dropna().unique()]
        )
        selected_grades = st.multiselect(
            "Grados",
            grades,
            default=grades,
            format_func=lambda g: f"{g}°",
            key=f"grades_{test_name}",
        )

    if selected_sites:
        d = d[d["Sede"].astype(str).isin(selected_sites)]
    if selected_grades:
        d = d[d["Grado_num"].isin(selected_grades)]

    if d.empty:
        st.warning("Los filtros seleccionados no dejan datos.")
        return

    render_main_line(d, aggregated=aggregated)

    if not aggregated:
        with st.expander(
            "Ver distribución por los cuatro niveles de desempeño",
            expanded=False,
        ):
            render_level_distribution(d)

        with st.expander(
            "Ver patrones por colegio de origen",
            expanded=False,
        ):
            render_origin_patterns(d)
    else:
        st.info(
            "La base precargada es agregada. "
            "Carga el Excel completo para habilitar niveles individuales "
            "y patrones por colegio de origen."
        )


st.title("📊 Estado de llegada 2026")
st.caption(
    "Cinco hojas independientes. Cada una analiza una prueba específica; "
    "no se muestran resultados generales mezclando pruebas."
)

with st.expander("Criterios de lectura", expanded=False):
    st.markdown(
        """
- **Año fijo:** 2026.
- **Progreso limitado:** 0% a 25%.
- **Emergente:** >25% a 50%.
- **En aceleración:** >50% a 75%.
- **Avanzado:** >75% a 100%.
- La visual principal cuenta estudiantes con **≤50% de aciertos**.
- Los rótulos de prueba no incluyen el grado.
- Ciencias sociales integra Competencias ciudadanas, Pensamiento ciudadano y Sociales y ciudadanas.
        """
    )

uploaded = st.file_uploader(
    "Cargar Excel completo (opcional)",
    type=["xlsx", "xls"],
    help=(
        "Sin archivo, el dashboard usa la base agregada precargada. "
        "Con el Excel completo se habilitan patrones por colegio de origen."
    ),
)

aggregated = uploaded is None

if uploaded is not None:
    try:
        raw = read_excel(uploaded.getvalue())
        data = process_raw(raw)
    except Exception as exc:
        st.error(f"No fue posible procesar el Excel: {exc}")
        st.stop()
else:
    data = read_default_summary()
    if data.empty:
        st.info("Carga el Excel para visualizar los resultados.")
        st.stop()

tabs = st.tabs(TESTS)

for tab, test_name in zip(tabs, TESTS):
    with tab:
        render_test_page(
            data=data,
            test_name=test_name,
            aggregated=aggregated,
        )
