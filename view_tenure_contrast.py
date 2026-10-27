"""Comparación descriptiva EFA 2026 entre antigüedad <=1 y >=3 años.

Se trabaja con registros individuales cargados en sesión; no se publica la
información identificable ni se infiere >=3 a partir de agregados generales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    LEVEL_ORDER,
    MIN_PUBLIC_GROUP_SIZE,
    SITES,
    TESTS,
    YEAR,
)

GROUP_YOUNG = "≤1 año"
GROUP_OLD = "≥3 años"
GROUPS = [GROUP_YOUNG, GROUP_OLD]
GROUP_COLORS = {GROUP_YOUNG: "#1976B9", GROUP_OLD: "#E19A32"}


def _student_id_key(values: pd.Series) -> pd.Series:
    """Normaliza IDs que Excel importa como números terminados en .0."""
    return (
        values.astype("string").str.strip()
        .str.replace(r"\\.0+$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    )


def _prepare_tenure_roster(tenure_roster: pd.DataFrame) -> pd.DataFrame:
    """Acepta el CSV de permanencia, sin conservar nombres ni correos."""
    required = {"OrgDefinedId", "AntiguedadBS", "Name"}
    missing = required - set(tenure_roster.columns)
    if missing:
        raise ValueError(
            "Al archivo de permanencia le faltan columnas: "
            + ", ".join(sorted(missing))
        )
    codes = {
        "MOSQUERA": "MOS",
        "COTA": "COT",
        "ZIPAQUIRA": "ZIPA",
        "TUNJA": "TUN",
        "PORTAL DE GENOVES": "BAQ",
        "USAQUEN 170": "USAQ",
        "USAQUEN": "USAQ",
    }
    import unicodedata

    def norm_site(value):
        name = "".join(
            ch for ch in unicodedata.normalize("NFKD", str(value).upper())
            if not unicodedata.combining(ch)
        ).strip()
        return codes.get(name, name)

    result = pd.DataFrame({
        "_student_key": _student_id_key(tenure_roster["OrgDefinedId"]),
        "_tenure_site": tenure_roster["Name"].map(norm_site),
        "_tenure_years": pd.to_numeric(
            tenure_roster["AntiguedadBS"], errors="coerce"
        ),
    })
    result = result.dropna(subset=["_student_key"])
    duplicates = result["_student_key"].duplicated(keep=False)
    if duplicates.any():
        conflicting = result[duplicates].groupby("_student_key").agg(
            sites=("_tenure_site", "nunique"),
            years=("_tenure_years", "nunique"),
        )
        if (conflicting["sites"].gt(1) | conflicting["years"].gt(1)).any():
            raise ValueError(
                "Existen identificadores duplicados con antigüedades "
                "o sedes contradictorias en el archivo."
            )
        result = result.drop_duplicates("_student_key", keep="first")
    return result


def _tenure_observations(
    attempts: pd.DataFrame,
    tenure_roster: pd.DataFrame,
) -> pd.DataFrame:
    """Cruce por identificador, validando sede y usando antigüedad individual."""
    required = {
        "Sede", "Grado_num", "Prueba", "IdentiEstudiante",
        "Porcentaje_acierto",
    }
    missing = required - set(attempts.columns)
    if missing:
        raise ValueError(
            "La base EFA no contiene: " + ", ".join(sorted(missing))
        )

    data = attempts.copy()
    data["_student_key"] = _student_id_key(data["IdentiEstudiante"])
    data["Sede"] = data["Sede"].astype(str).str.strip().str.upper()
    data["Grado_num"] = pd.to_numeric(data["Grado_num"], errors="coerce")
    data["Porcentaje_acierto"] = pd.to_numeric(
        data["Porcentaje_acierto"], errors="coerce"
    )
    if "Año" in data.columns:
        data = data[pd.to_numeric(data["Año"], errors="coerce") == YEAR]
    data = data[
        data["Sede"].isin(SITES)
        & data["Prueba"].isin(TESTS)
        & data["_student_key"].notna()
        & data["Grado_num"].between(3, 11)
        & data["Porcentaje_acierto"].between(0, 100)
    ].copy()

    efa_ids = data["_student_key"].nunique()
    roster = _prepare_tenure_roster(tenure_roster)
    data = data.merge(
        roster, on="_student_key", how="left",
        validate="many_to_one", indicator="_merge_tenure",
    )
    matched_ids = data.loc[
        data["_merge_tenure"] == "both", "_student_key"
    ].nunique()
    mismatched_ids = data.loc[
        (data["_merge_tenure"] == "both")
        & (data["Sede"] != data["_tenure_site"]), "_student_key"
    ].nunique()
    data = data[
        (data["_merge_tenure"] == "both")
        & (data["Sede"] == data["_tenure_site"])
    ].copy()
    years = data["_tenure_years"]
    data["Grupo"] = np.select(
        [years.ge(0) & years.le(1), years.ge(3)],
        [GROUP_YOUNG, GROUP_OLD],
        default="Sin grupo",
    )
    data = data[data["Grupo"].isin(GROUPS)].copy()

    if "TimeCompleted" in data.columns:
        data["_orden_fecha"] = pd.to_datetime(
            data["TimeCompleted"], errors="coerce"
        )
        data = data.sort_values("_orden_fecha", kind="stable")

    data = data.drop_duplicates(
        ["_student_key", "Sede", "Grado_num", "Prueba"],
        keep="last",
    )
    data["Nivel"] = pd.cut(
        data["Porcentaje_acierto"],
        bins=[-0.001, 25, 50, 75, 100],
        labels=LEVEL_ORDER,
        include_lowest=True,
    )
    data["Nivel_3_4"] = data["Porcentaje_acierto"] > 50
    data.attrs["efa_ids"] = efa_ids
    data.attrs["matched_ids"] = matched_ids
    data.attrs["mismatched_ids"] = mismatched_ids
    return data


def _summarize(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby(
            ["Sede", "Grado_num", "Prueba", "Grupo"],
            as_index=False, observed=True,
        )
        .agg(
            Estudiantes=("_student_key", "nunique"),
            Puntaje=("Porcentaje_acierto", "mean"),
            En_3_4=("Nivel_3_4", "sum"),
        )
    )
    grouped["Pct_3_4"] = (
        grouped["En_3_4"] / grouped["Estudiantes"] * 100
    )
    return grouped


def render_tenure_contrast(
    attempts: pd.DataFrame,
    aggregated: bool,
):
    st.markdown("## Desempeño EFA · permanencia en el modelo Innova")
    st.caption(
        "Contraste descriptivo de estudiantes con un año o menos y con "
        "tres años o más de permanencia. Los estudiantes con 2 años "
        "no se incluyen en ninguno de los dos grupos."
    )

    roster_upload = st.file_uploader(
        "Cargar antigüedad individual de estudiantes (CSV)",
        type=["csv"], key="tenure_individual_roster",
        help=(
            "Selecciona el archivo con OrgDefinedId, AntiguedadBS y Name. "
            "Los identificadores se usan solo para unir registros en la sesión."
        ),
    )
    if aggregated:
        st.info(
            "Para comparar los grupos necesitas cargar también el Excel "
            "original de EFA en el control «Cargar Excel completo» al "
            "inicio de la aplicación. Los agregados precargados no incluyen "
            "identificadores para cruzar ambas bases."
        )
        return
    if roster_upload is None:
        st.info(
            "Carga aquí el CSV de antigüedad para emparejarlo con los "
            "resultados de EFA mediante OrgDefinedId e IdentiEstudiante."
        )
        return

    try:
        tenure_roster = pd.read_csv(
            roster_upload, dtype="string", encoding="utf-8-sig"
        )
        data = _tenure_observations(attempts, tenure_roster)
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"No fue posible cruzar EFA y antigüedad: {exc}")
        return

    efa_ids = int(data.attrs.get("efa_ids", 0))
    matched_ids = int(data.attrs.get("matched_ids", 0))
    mismatch_ids = int(data.attrs.get("mismatched_ids", 0))
    st.caption(
        f"Cruce por identificador: {matched_ids:,} de {efa_ids:,} "
        "estudiantes EFA encontrados en el archivo de permanencia. "
        f"Se excluyeron {mismatch_ids:,} identificadores con sedes "
        "distintas. Los demás no emparejados no se clasifican."
    )
    st.caption(
        "Se utiliza exclusivamente AntiguedadBS de cada estudiante. "
        "La pertenencia a ≤1 o ≥3 años no se asigna automáticamente por sede. "
        "Quedan excluidos los estudiantes de 2 años."
    )
    if data.empty:
        st.warning(
            "No hay estudiantes de EFA con antigüedad individual válida "
            "emparejada y sede coincidente para los dos grupos."
        )
        return

    eligible_sites = [
        site for site in SITES
        if set(data.loc[data["Sede"] == site, "Grupo"]) == set(GROUPS)
    ]

    test_options = [
        test for test in TESTS if test in set(data["Prueba"])
    ]
    default_test = (
        "Matemáticas" if "Matemáticas" in test_options
        else test_options[0]
    )
    a, b, c = st.columns([1.1, 1.3, 1.8])
    with a:
        test = st.selectbox(
            "Prueba", test_options,
            index=test_options.index(default_test),
            key="tenure_test",
        )
    with b:
        sites = st.multiselect(
            "Sedes comparables", eligible_sites,
            default=eligible_sites,
            key="tenure_sites",
            help="Solo sedes con observaciones de ambos grupos.",
        )
    available_grades = sorted(
        data.loc[data["Prueba"] == test, "Grado_num"]
        .dropna().astype(int).unique().tolist()
    )
    with c:
        grades = st.multiselect(
            "Grados", available_grades,
            default=available_grades,
            format_func=lambda value: f"{value}°",
            key="tenure_grades",
        )
    if not sites or not grades:
        st.info(
            "No hay sedes con ambos grupos para comparar, o no se han "
            "seleccionado sedes y grados. Revisa los filtros."
        )
        return

    selected = data[
        data["Prueba"].eq(test)
        & data["Sede"].isin(sites)
        & data["Grado_num"].isin(grades)
    ].copy()
    summary = _summarize(selected)
    if summary.empty:
        st.warning("No hay registros para esa combinación de filtros.")
        return

    # Comparar solo celdas con n suficiente en LOS DOS grupos, a nivel
    # sede × grado × prueba. No comparar cohortes con tamaños insuficientes.
    n_by_group = summary.pivot_table(
        index=["Sede", "Grado_num"],
        columns="Grupo",
        values="Estudiantes",
        aggfunc="first",
    ).reindex(columns=GROUPS)
    valid_cells = n_by_group[
        n_by_group[GROUP_YOUNG].ge(MIN_PUBLIC_GROUP_SIZE)
        & n_by_group[GROUP_OLD].ge(MIN_PUBLIC_GROUP_SIZE)
    ].index
    summary = summary.set_index(["Sede", "Grado_num"])
    summary = summary.loc[summary.index.isin(valid_cells)].reset_index()
    selected = selected.set_index(["Sede", "Grado_num"])
    selected = selected.loc[selected.index.isin(valid_cells)].reset_index()
    if summary.empty:
        st.warning(
            "Ninguna combinación sede × grado presenta al menos "
            f"{MIN_PUBLIC_GROUP_SIZE} estudiantes en ambos grupos."
        )
        return

    total_cells = len(n_by_group)
    retained_cells = len(valid_cells)
    total_n = selected.groupby("Grupo")["_student_key"].nunique()
    m1, m2, m3 = st.columns(3)
    m1.metric("Estudiantes ≤1 año", int(total_n.get(GROUP_YOUNG, 0)))
    m2.metric("Estudiantes ≥3 años", int(total_n.get(GROUP_OLD, 0)))
    m3.metric("Sede × grado comparables", f"{retained_cells} de {total_cells}")
    st.caption(
        f"Solo se muestran comparaciones con al menos "
        f"{MIN_PUBLIC_GROUP_SIZE} estudiantes por grupo en el mismo grado "
        "y sede. Los conteos globales son estudiantes únicos dentro "
        "de las celdas comparables, no el total de toda la red."
    )

    summary["Grado"] = summary["Grado_num"].astype(int).astype(str) + "°"
    site_order = [
        site for site in SITES if site in set(summary["Sede"])
    ]

    for value_col, title, y_label in [
        ("Puntaje", "Puntaje promedio en Evaluar para Avanzar",
         "Puntaje promedio (%)"),
        ("Pct_3_4", "Proporción en niveles 3 + 4",
         "Estudiantes en niveles 3 + 4 (%)"),
    ]:
        st.markdown(f"### {title}")
        chart = px.bar(
            summary, x="Grado", y=value_col,
            color="Grupo", barmode="group",
            facet_col="Sede", facet_col_wrap=2,
            category_orders={
                "Sede": site_order, "Grupo": GROUPS,
                "Grado": [f"{g}°" for g in sorted(grades)],
            },
            color_discrete_map=GROUP_COLORS,
            custom_data=["Estudiantes"],
            labels={
                "Grado": "Grado", value_col: y_label,
                "Grupo": "Antigüedad",
            },
        )
        chart.update_yaxes(range=[0, 105], ticksuffix="%")
        chart.update_traces(
            hovertemplate=(
                "Grado %{x}<br>%{fullData.name}: %{y:.1f}%"
                "<br>Estudiantes: %{customdata[0]:.0f}"
                "<extra></extra>"
            ),
        )
        chart.for_each_annotation(
            lambda annotation: annotation.update(
                text=annotation.text.replace("Sede=", "")
            )
        )
        chart.update_layout(
            height=max(430, 340 * ((len(site_order) + 1) // 2)),
            legend=dict(orientation="h", x=0, y=1.10),
            margin=dict(l=35, r=15, t=80, b=45),
        )
        st.plotly_chart(
            chart, use_container_width=True,
            key=f"tenure_{value_col}",
        )

    gap = summary.pivot_table(
        index=["Sede", "Grado_num"],
        columns="Grupo",
        values=["Puntaje", "Pct_3_4", "Estudiantes"],
        aggfunc="first",
    )
    table = pd.DataFrame({
        "Sede": gap.index.get_level_values("Sede"),
        "Grado": gap.index.get_level_values("Grado_num").astype(int),
        "n ≤1": gap[("Estudiantes", GROUP_YOUNG)].astype(int).values,
        "n ≥3": gap[("Estudiantes", GROUP_OLD)].astype(int).values,
        "EFA ≤1 (%)": gap[("Puntaje", GROUP_YOUNG)].round(1).values,
        "EFA ≥3 (%)": gap[("Puntaje", GROUP_OLD)].round(1).values,
        "Brecha EFA (pp)": (
            gap[("Puntaje", GROUP_OLD)] -
            gap[("Puntaje", GROUP_YOUNG)]
        ).round(1).values,
        "Brecha nivel 3+4 (pp)": (
            gap[("Pct_3_4", GROUP_OLD)] -
            gap[("Pct_3_4", GROUP_YOUNG)]
        ).round(1).values,
    })
    st.markdown("### Tabla de contraste por sede y grado")
    st.dataframe(
        table.sort_values(["Sede", "Grado"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Las brechas se calculan como ≥3 años menos ≤1 año, en puntos "
        "porcentuales. Un valor positivo indica mayor resultado observado "
        "en el grupo de ≥3 años, no evidencia de valor agregado ni de "
        "un efecto causal del modelo: los grupos tienen composiciones "
        "y trayectorias potencialmente distintas."
    )
