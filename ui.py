
from __future__ import annotations

import pandas as pd
import streamlit as st


LEVEL_ORDER = [
    "Progreso limitado",
    "Emergente",
    "En aceleración",
    "Avanzado",
]

LEVEL_COLORS = {
    "Progreso limitado": "#C0392B",
    "Emergente": "#E67E22",
    "En aceleración": "#F1C40F",
    "Avanzado": "#27AE60",
}


def multiselect_filter(label: str, data: pd.DataFrame, column: str, key: str):
    if column not in data.columns:
        return []
    values = (
        data[column]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return st.multiselect(label, values, key=key)


def apply_filters(
    df: pd.DataFrame,
    sedes: list[str],
    grados: list[str],
    areas: list[str],
    pruebas: list[str],
    cursos: list[str],
):
    out = df.copy()
    if sedes:
        out = out[out["Sede"].astype(str).isin(sedes)]
    if grados:
        out = out[out["Grado"].astype(str).isin(grados)]
    if areas:
        out = out[out["Area"].astype(str).isin(areas)]
    if pruebas:
        out = out[out["QuizName"].astype(str).isin(pruebas)]
    if cursos and "Curso" in out.columns:
        out = out[out["Curso"].astype(str).isin(cursos)]
    return out


def dataframe_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )
