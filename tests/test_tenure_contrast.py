"""Pruebas sin datos personales para el cruce EFA × antigüedad."""
import pandas as pd
import pytest

from view_tenure_contrast import (
    _prepare_tenure_roster,
    _student_id_key,
    _tenure_observations,
)


def test_ids_from_excel_match_roster():
    values = pd.Series(["1001.0", " 1002 ", "001003", None])
    result = _student_id_key(values)
    assert result.iloc[:3].tolist() == ["1001", "1002", "001003"]
    assert pd.isna(result.iloc[3])


def test_tenure_uses_actual_years_even_for_new_schools():
    attempts = pd.DataFrame({
        "IdentiEstudiante": [
            "1001.0", "1001.0", "1002", "1003", "1004", "1005", "1006"
        ],
        "Sede": ["MOS", "MOS", "MOS", "MOS", "USAQ", "MOS", "MOS"],
        "Grado_num": [5] * 7,
        "Prueba": ["Matemáticas"] * 7,
        "Porcentaje_acierto": [50, 60, 80, 45, 70, 75, 40],
        "Año": [2026] * 7,
    })
    roster = pd.DataFrame({
        "OrgDefinedId": [
            "1001", "1002", "1003", "1004", "1005", "1006"
        ],
        "Name": [
            "MOSQUERA", "MOSQUERA", "MOSQUERA",
            "USAQUÉN 170", "COTA", "MOSQUERA"
        ],
        "AntiguedadBS": ["0", "5", "2", "4", "1", "-1"],
    })
    result = _tenure_observations(attempts, roster)
    grouped = result.set_index("_student_key")["Grupo"].to_dict()
    assert grouped == {
        "1001": "≤1 año", "1002": "≥3 años", "1004": "≥3 años",
    }
    assert result.loc[
        result["_student_key"] == "1001", "Porcentaje_acierto"
    ].iloc[0] == 60
    assert result.attrs["mismatched_ids"] == 1


def test_duplicate_conflicting_roster_ids_rejected():
    roster = pd.DataFrame({
        "OrgDefinedId": ["1", "1"],
        "Name": ["MOSQUERA", "MOSQUERA"],
        "AntiguedadBS": [0, 3],
    })
    with pytest.raises(ValueError, match="contradictorias"):
        _prepare_tenure_roster(roster)
