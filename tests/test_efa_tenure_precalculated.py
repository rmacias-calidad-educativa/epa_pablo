"""Auditar consistencia y supresión de datos públicos EFA por antigüedad."""
import pandas as pd
from pathlib import Path


def test_public_efa_tenure_aggregates_are_anonymous_and_comparable():
    path = Path(__file__).resolve().parents[1] / "data/efa_tenure_comparison_2026.csv"
    d = pd.read_csv(path)
    assert not d.empty
    forbidden = {"IdentiEstudiante", "OrgDefinedId", "Nombre", "Apellido",
                 "ExternalEmail", "Documento", "Email", "Pregunta"}
    assert not forbidden.intersection(d.columns)
    levels = ["Nivel_1", "Nivel_2", "Nivel_3", "Nivel_4"]
    assert d[levels].sum(axis=1).eq(d["Estudiantes"]).all()
    assert d["Estudiantes"].ge(5).all()
    assert d["Suma_puntaje"].between(0, d["Estudiantes"] * 100).all()
    keys = ["Sede", "Grado_num", "Prueba"]
    assert d.groupby(keys)["Grupo"].nunique().eq(2).all()
    assert d.groupby(keys).size().eq(2).all()
    assert set(d["Grupo"]) == {"≤1 año", "≥3 años"}


def test_precalculated_efa_loader_derives_correct_percentages():
    from view_tenure_precalculated import read_efa_tenure_aggregate
    d = read_efa_tenure_aggregate()
    assert not d.empty
    assert d["Puntaje"].between(0, 100).all()
    assert d["Niveles_3_4"].between(0, 100).all()
    assert d["Sede"].isin(["MOS"]).all()
