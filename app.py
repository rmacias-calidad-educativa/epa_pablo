
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.processing import aggregate_group, build_attempt_table
from src.ui import (
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
    st.info(
        "Carga el Excel para activar el dashboard. El repositorio no necesita guardar "
        "datos personales: el archivo se procesa durante la sesión."
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
        "Procedencia",
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

# ------------------------- Procedencia -------------------------
with tabs[3]:
    st.subheader("Antigüedad en BS")
    if "AntiguedadBS" in filtered.columns and filtered["AntiguedadBS"].notna().any():
        seniority = aggregate_group(filtered, ["AntiguedadBS"]).sort_values("AntiguedadBS")
        fig = px.bar(
            seniority,
            x="AntiguedadBS",
            y="Promedio",
            text=seniority["Promedio"].map(lambda x: f"{x:.1f}%"),
            hover_data=["Estudiantes", "Requiere_apoyo", "Cobertura_promedio"],
            title="Desempeño según antigüedad BS",
            labels={"AntiguedadBS": "Antigüedad BS", "Promedio": "% acierto"},
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Colegio de origen")
    origin = filtered[
        filtered["colegio_de_origen"].notna()
        & (filtered["colegio_de_origen"].astype(str).str.strip() != "")
    ].copy()

    if origin.empty:
        st.info("No hay información de colegio de origen en el filtro actual.")
    else:
        min_students = st.slider(
            "Mínimo de estudiantes por colegio para comparar",
            min_value=1,
            max_value=20,
            value=3,
        )
        origin_summary = aggregate_group(origin, ["colegio_de_origen"])
        origin_summary = origin_summary[origin_summary["Estudiantes"] >= min_students]
        origin_summary = origin_summary.sort_values(
            ["Requiere_apoyo", "Estudiantes"],
            ascending=[False, False],
        )

        st.caption(
            "La tabla permite ubicar concentraciones de reto por procedencia. "
            "El filtro mínimo evita interpretar colegios representados por muy pocos estudiantes."
        )
        st.dataframe(
            origin_summary.head(50),
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
