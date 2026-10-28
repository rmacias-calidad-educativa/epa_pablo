"""Comprobaciones de la referencia RED para dimensiones EFA."""
import pandas as pd
import pytest

from view_general import (
    dimension_site_vs_network,
    dimension_summary_by_grade,
)


def test_network_is_weighted_and_not_limited_to_selected_site():
    dimensions = pd.DataFrame({
        "Sede": ["USAQ", "MOS", "USAQ", "MOS"],
        "Grado_num": [5, 5, 6, 6],
        "Dimension": ["Comunicación"] * 4,
        "Estudiantes": [10, 90, 10, 90],
        "Promedio_dimension": [80.0, 50.0, 60.0, 70.0],
    })
    site, red = dimension_site_vs_network(dimensions, "USAQ", aggregated=True)
    assert site["Promedio_dimension"].tolist() == [80.0, 60.0]
    assert red["Promedio_dimension"].tolist() == [53.0, 69.0]
    assert red["Estudiantes"].tolist() == [100, 100]


def test_raw_dimension_reference_same_grade_and_dimension():
    data = pd.DataFrame({
        "Sede": ["USAQ", "USAQ", "MOS", "MOS", "MOS"],
        "Grado_num": [5, 5, 5, 5, 6],
        "Dimension": ["Razonamiento"] * 5,
        "IdentiEstudiante": ["A", "B", "C", "D", "E"],
        "Porcentaje_dimension": [60, 80, 40, 100, 20],
    })
    site, red = dimension_site_vs_network(data, "USAQ", aggregated=False)
    assert len(site) == 1
    assert site.iloc[0]["Promedio_dimension"] == pytest.approx(70)
    assert red.loc[red["Grado_num"] == 5, "Promedio_dimension"].iat[0] == pytest.approx(70)
    assert red.loc[red["Grado_num"] == 6, "Promedio_dimension"].iat[0] == 20


def test_empty_dimension_summary_is_safe():
    result = dimension_summary_by_grade(pd.DataFrame(), aggregated=True)
    assert result.empty
    assert "Promedio_dimension" in result.columns
