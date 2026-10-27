"""Distribución descriptiva de evaluación interna T1 y T2.

Los agregados disponibles están cruzados con EFA y NO contienen sede ni
campo de aprendizaje. No se atribuyen estos datos a sedes específicas.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

LEVELS = [
    "Progreso limitado", "Emergente", "En aceleración", "Avanzado",
]
COLORS = {
    "Progreso limitado": "#C93C3C",
    "Emergente": "#E9A529",
    "En aceleración": "#93C354",
    "Avanzado": "#2877BC",
}
TRIMESTERS = ["TRIMESTRE 1", "TRIMESTRE 2"]


GRADE_LABELS = [
    "Prejardín", "Jardín", "Transición",
    *[str(i) for i in range(1, 12)],
]
GRADE_ALIASES = {
    "PREJARDIN": "Prejardín", "PRE-JARDIN": "Prejardín",
    "JARDIN": "Jardín", "TRANSICION": "Transición",
    "PREKINDER": "Prejardín", "KINDER": "Jardín",
}
LEVEL_ALIASES = {
    "PROGRESO LIMITADO": "Progreso limitado",
    "EMERGENTE": "Emergente",
    "EN ACELERACION": "En aceleración",
    "AVANZADO": "Avanzado",
}
SITE_NAMES = {
    "COT": "Cota", "MOS": "Mosquera", "BAQ": "Portal de Genovés",
    "USAQ": "Usaquén", "TUN": "Tunja", "ZIPA": "Zipaquirá",
}


def _normalize_grade(value):
    import re
    import unicodedata
    raw = str(value).strip().upper()
    norm = "".join(ch for ch in unicodedata.normalize("NFKD", raw)
                   if not unicodedata.combining(ch))
    norm = norm.replace(" ", "").replace("°", "")
    if norm in {"PREJARDIN", "PRE-JARDIN", "PREKINDER"}:
        return "Prejardín"
    if norm in {"JARDIN", "KINDER"}:
        return "Jardín"
    if norm in {"TRANSICION", "GRADO0", "0"}:
        return "Transición"
    match = re.search(r"\d+", norm)
    if match and 1 <= int(match.group()) <= 11:
        return str(int(match.group()))
    words = {
        "PRIMERO": "1", "SEGUNDO": "2", "TERCERO": "3",
        "CUARTO": "4", "QUINTO": "5", "SEXTO": "6",
        "SEPTIMO": "7", "OCTAVO": "8", "NOVENO": "9",
        "DECIMO": "10", "UNDECIMO": "11", "ONCE": "11",
    }
    return words.get(norm)


def _normalize_token(value):
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize(
        "NFKD", str(value).strip().upper()
    ) if not unicodedata.combining(ch))


def _prepare_internal_source(frame):
    """Acepta registros individuales o conteos agrupados; sin inferir datos."""
    names = {
        "sede": "Sede", "colegio": "Sede",
        "grado": "Grado", "grado_num": "Grado",
        "campo": "Campo", "campo_de_aprendizaje": "Campo",
        "campo de aprendizaje": "Campo", "area": "Campo",
        "trimestre": "Trimestre", "periodo": "Trimestre",
        "nivel": "Nivel", "nivel_de_logro": "Nivel",
        "desempeno": "Nivel", "desempeño": "Nivel",
        "escala despues de potenciacion": "Nivel",
        "estudiantes": "Estudiantes", "conteo": "Estudiantes",
        "n": "Estudiantes",
    }
    df = frame.rename(columns={
        col: names.get(_normalize_token(col).lower(), col)
        for col in frame.columns
    }).copy()
    required = {"Sede", "Grado", "Campo", "Trimestre", "Nivel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(missing)))
    df["Grado"] = df["Grado"].map(_normalize_grade)
    df["Trimestre"] = df["Trimestre"].map(
        lambda v: "TRIMESTRE 1" if _normalize_token(v) in
        {"T1", "1", "TRIMESTRE 1", "TRIMESTRE1"} else (
            "TRIMESTRE 2" if _normalize_token(v) in
            {"T2", "2", "TRIMESTRE 2", "TRIMESTRE2"} else None)
    )
    df["Nivel"] = df["Nivel"].map(
        lambda v: LEVEL_ALIASES.get(_normalize_token(v))
    )
    df["Sede"] = df["Sede"].map(
        lambda v: SITE_NAMES.get(_normalize_token(v), str(v).strip())
    )
    # Con datos individuales, contar una sola vez por estudiante,
    # sede, trimestre, grado y campo antes de agregar.
    if "Documento" in df.columns and "Estudiantes" not in df.columns:
        df = df.drop_duplicates(
            subset=["Sede", "Trimestre", "Grado", "Campo", "Documento"]
        )
    if "Estudiantes" in df.columns:
        df["Estudiantes"] = pd.to_numeric(
            df["Estudiantes"], errors="coerce"
        ).fillna(0)
    else:
        df["Estudiantes"] = 1
    return df[
        df["Grado"].isin(GRADE_LABELS)
        & df["Trimestre"].isin(TRIMESTERS)
        & df["Nivel"].isin(LEVELS)
        & df["Estudiantes"].gt(0)
    ].copy()


def _render_complete_internal(frame):
    st.markdown("## Evaluación interna · todas las sedes y grados")
    st.caption(
        "Distribuciones por sede, campo de aprendizaje y grado, "
        "desde prejardín hasta 11.º, para T1 y T2."
    )
    d = _prepare_internal_source(frame)
    if d.empty:
        st.warning("El archivo no contiene registros compatibles para T1 y T2.")
        return
    available_sites = sorted(d["Sede"].unique().tolist())
    available_grades = [g for g in GRADE_LABELS if g in set(d["Grado"])]
    available_fields = sorted(d["Campo"].astype(str).unique().tolist())
    a, b, c = st.columns(3)
    with a:
        sites = st.multiselect("Sedes", available_sites,
                               default=available_sites, key="int_sites")
    with b:
        grades = st.multiselect("Grados", GRADE_LABELS,
                                default=available_grades, key="int_grades")
    with c:
        fields = st.multiselect("Campos de aprendizaje", available_fields,
                                default=available_fields, key="int_fields")
    d = d[d["Sede"].isin(sites) & d["Grado"].isin(grades)
          & d["Campo"].isin(fields)].copy()
    if d.empty:
        st.info("No hay registros en la selección.")
        return
    # Los grados se seleccionan con el filtro, pero las barras corresponden
    # exclusivamente a campos de aprendizaje dentro de cada sede.
    d = d.groupby(
        ["Sede", "Campo", "Trimestre", "Nivel"], as_index=False
    )["Estudiantes"].sum()
    # Comparación simultánea: paneles de sedes en una misma figura.
    # Cada barra es un Área, sin combinar su nombre con el grado.
    for trimester in TRIMESTERS:
        st.markdown(
            "### " + ("Trimestre 1" if trimester.endswith("1")
                      else "Trimestre 2")
        )
        subset = d[d["Trimestre"] == trimester].copy()
        present_sites = [
            site for site in sites
            if not subset[subset["Sede"] == site].empty
        ]
        if not present_sites:
            st.info("No hay registros en este trimestre.")
            continue
        columns = min(3, len(present_sites))
        rows = (len(present_sites) + columns - 1) // columns
        fig = make_subplots(
            rows=rows, cols=columns,
            subplot_titles=[site.upper() for site in present_sites],
            horizontal_spacing=0.13,
            vertical_spacing=0.13 if rows > 1 else 0.06,
        )
        max_areas = 0
        for index, site in enumerate(present_sites):
            row_no, col_no = divmod(index, columns)
            site_df = subset[subset["Sede"] == site].copy()
            site_df["Total"] = site_df.groupby("Campo")[
                "Estudiantes"
            ].transform("sum")
            site_df["Porcentaje"] = (
                site_df["Estudiantes"] / site_df["Total"] * 100
            )
            order = sorted(site_df["Campo"].astype(str).unique().tolist())
            max_areas = max(max_areas, len(order))
            for level in LEVELS:
                part = site_df[site_df["Nivel"] == level].set_index("Campo")
                part = part.reindex(order)
                vals = part["Porcentaje"].fillna(0).tolist()
                counts = part["Estudiantes"].fillna(0).tolist()
                totals = part["Total"].fillna(0).tolist()
                fig.add_trace(
                    go.Bar(
                        x=vals, y=order,
                        name=level,
                        orientation="h",
                        marker_color=COLORS[level],
                        text=[
                            f"{value:.1f}%" if value >= 8 else ""
                            for value in vals
                        ],
                        textposition="inside",
                        customdata=list(zip(counts, totals)),
                        hovertemplate=(
                            "%{y}<br>%{fullData.name}: %{x:.1f}%"
                            "<br>n=%{customdata[0]:.0f}"
                            " de %{customdata[1]:.0f}"
                            "<extra></extra>"
                        ),
                        showlegend=index == 0,
                    ),
                    row=row_no + 1, col=col_no + 1,
                )
            fig.update_xaxes(
                range=[0, 100], ticksuffix="%",
                row=row_no + 1, col=col_no + 1,
            )
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=list(reversed(order)),
                tickfont=dict(size=9),
                row=row_no + 1, col=col_no + 1,
            )
        fig.update_layout(
            barmode="stack",
            height=rows * max(440, max_areas * 36 + 140),
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=15, r=15, t=100, b=35),
        )
        st.plotly_chart(
            fig, use_container_width=True,
            key=f"int_sites_compare_{trimester}",
        )
    st.caption(
        "Cada barra corresponde a un campo de aprendizaje de la sede. "
        "Los grados seleccionados se agregan dentro de cada campo; "
        "una barra no representa necesariamente estudiantes únicos entre grados."
    )


def render_internal_distribution_only(distribution: pd.DataFrame):
    uploaded = st.file_uploader(
        "Cargar Boletines 2026 por sede y campo (CSV o Excel)",
        type=["csv", "xlsx"], key="internal_complete_upload",
        help=("Admite el archivo Boletines 2026 con columnas Sede, Grado, "
              "Área, Trimestre y Escala DESPUÉS DE POTENCIACIÓN, "
              "o una tabla agregada con Campo, Nivel y Estudiantes.")
    )
    if uploaded is not None:
        try:
            full = (
                pd.read_csv(uploaded)
                if uploaded.name.lower().endswith(".csv")
                else pd.read_excel(uploaded)
            )
            _render_complete_internal(full)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"No fue posible procesar la fuente: {exc}")
        return
    st.info(
        "Para mostrar prejardín a 11.º por sede y campo, carga aquí "
        "la evaluación interna completa. Mientras tanto se muestran "
        "solo los cruces disponibles de EFA, que no incluyen las sedes."
    )

    st.markdown("## Evaluación interna · distribución del desempeño")
    st.caption(
        "Comparación de T1 y T2 en barras apiladas al 100 %, por prueba EFA "
        "y grado. Cada distribución procede de estudiantes emparejados con EFA."
    )
    st.warning(
        "La base precargada de este módulo no tiene campos de aprendizaje "
        "ni sedes. Por eso esta vista utiliza únicamente las pruebas y los "
        "grados disponibles. No equivale a la distribución de todos los "
        "estudiantes de cada sede mostrada en la imagen de referencia."
    )

    if distribution is None or distribution.empty:
        st.info("No se encontraron datos de evaluación interna.")
        return

    base = distribution[
        distribution["Trimestre"].astype(str).isin(TRIMESTERS)
    ].copy()
    base["Estudiantes"] = pd.to_numeric(
        base["Estudiantes"], errors="coerce"
    ).fillna(0)
    base["Grado_num"] = pd.to_numeric(
        base["Grado_num"], errors="coerce"
    )
    base = base.dropna(subset=["Grado_num"])
    if base.empty:
        st.info("No existen resultados de T1 y T2 para esta selección.")
        return

    tests = sorted(base["Prueba"].dropna().astype(str).unique().tolist())
    grades = sorted(base["Grado_num"].dropna().astype(int).unique().tolist())
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_tests = st.multiselect(
            "Pruebas", tests, default=tests,
            key="internal_only_tests",
        )
    with c2:
        selected_grades = st.multiselect(
            "Grados", grades, default=grades,
            format_func=lambda value: f"{value}°",
            key="internal_only_grades",
        )
    if not selected_tests or not selected_grades:
        st.info("Selecciona al menos una prueba y un grado.")
        return

    base = base[
        base["Prueba"].isin(selected_tests)
        & base["Grado_num"].astype(int).isin(selected_grades)
    ].copy()

    # Sumar conteos, no porcentajes ya calculados: cada fila de EFA
    # representa un estrato distinto dentro de grado/prueba/trimestre.
    base = (
        base.groupby(
            ["Trimestre", "Prueba", "Grado_num", "Interna_nivel"],
            as_index=False,
        )["Estudiantes"].sum()
    )
    base["Panel"] = (
        base["Prueba"].astype(str) + " · "
        + base["Grado_num"].astype(int).astype(str) + "°"
    )
    totals = base.groupby(
        ["Trimestre", "Panel"]
    )["Estudiantes"].transform("sum")
    base["Porcentaje"] = (
        base["Estudiantes"].div(totals.where(totals.ne(0))).mul(100)
    )
    base["Etiqueta"] = base["Porcentaje"].map(
        lambda v: f"{v:.1f}%" if pd.notna(v) and v >= 5 else ""
    )

    for trimester in TRIMESTERS:
        st.markdown(
            "### " + ("Trimestre 1" if trimester.endswith("1")
                      else "Trimestre 2")
        )
        subset = base[base["Trimestre"] == trimester].copy()
        if subset.empty:
            st.info("Sin registros para este trimestre.")
            continue

        panels = sorted(subset["Panel"].unique().tolist())
        chart = px.bar(
            subset,
            x="Porcentaje", y="Panel",
            color="Interna_nivel",
            orientation="h",
            barmode="stack",
            text="Etiqueta",
            custom_data=["Estudiantes"],
            color_discrete_map=COLORS,
            category_orders={
                "Interna_nivel": LEVELS,
                "Panel": list(reversed(panels)),
            },
            labels={
                "Porcentaje": "Porcentaje de estudiantes",
                "Panel": "Prueba y grado",
                "Interna_nivel": "Desempeño",
            },
        )
        chart.update_traces(
            textposition="inside",
            hovertemplate=(
                "%{y}<br>%{fullData.name}<br>"
                "Porcentaje: %{x:.1f}%<br>"
                "Estudiantes: %{customdata[0]:.0f}<extra></extra>"
            ),
        )
        chart.update_xaxes(range=[0, 100], ticksuffix="%")
        chart.update_layout(
            height=max(440, 36 * len(panels) + 150),
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=10, r=18, t=85, b=30),
        )
        st.plotly_chart(
            chart, use_container_width=True,
            key=f"internal_only_distribution_{trimester}",
        )

    st.caption(
        "Denominador: estudiantes con registro emparejado EFA–evaluación "
        "interna en cada prueba, grado y trimestre. Un estudiante puede "
        "aparecer en más de una prueba; no se suman las barras como "
        "personas únicas."
    )
