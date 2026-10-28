"""Regression tests: LATAM × EFA must not lose small score-frequency cells."""
import pytest

from config import MIN_PUBLIC_GROUP_SIZE
from data_quality import suppress_small_groups
from loaders import read_latam_efa_data
from view_latam import _spearman_from_aggregate, _weighted_group_average


def test_latam_total_counts_are_preserved_before_public_suppression():
    detail, summary = read_latam_efa_data()
    for area, expected in [("Matemáticas", 308), ("Lenguaje", 355)]:
        observed = detail[detail["Area"].eq(area)]["Estudiantes"].sum()
        total = int(summary.loc[
            summary["Area"].eq(area), "Emparejados_EFA_misma_area"
        ].iloc[0])
        assert observed == expected == total


def test_math_mos_selected_grades_include_all_matched_students():
    detail, _ = read_latam_efa_data()
    selected = detail[
        detail["Area"].eq("Matemáticas")
        & detail["Sede"].eq("MOS")
        & detail["Grado_num"].isin([5, 7, 8, 9])
    ]
    assert selected["Estudiantes"].sum() == 220
    assert _spearman_from_aggregate(selected) == pytest.approx(
        0.31767635, abs=1e-5
    )
    avg = selected.groupby("Categoria_LATAM").apply(
        lambda g: g["Estudiantes"].sum(), include_groups=False
    )
    assert avg.to_dict() == {
        "Avanzado": 29,
        "Básico": 61,
        "Inicio": 73,
        "Satisfactorio": 57,
    }


def test_privacy_threshold_applied_after_category_aggregation():
    detail, _ = read_latam_efa_data()
    selected = detail[
        detail["Area"].eq("Matemáticas")
        & detail["Sede"].eq("MOS")
        & detail["Grado_num"].isin([5, 7, 8, 9])
    ]
    rows = selected.groupby("Categoria_LATAM", as_index=False).apply(
        lambda x: __import__("pandas").Series({
            "n": int(x["Estudiantes"].sum()),
            "average": _weighted_group_average(x),
        }), include_groups=False,
    )
    publishable = suppress_small_groups(
        rows, "n", MIN_PUBLIC_GROUP_SIZE
    )
    assert publishable["n"].sum() == 220
    assert publishable["n"].ge(MIN_PUBLIC_GROUP_SIZE).all()
