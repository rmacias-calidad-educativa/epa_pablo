import pandas as pd

from processing import (
    build_attempt_table,
    expected_items,
    grade_number,
    normalize_area,
    performance_level,
)


def test_expected_items():
    assert expected_items("Matemáticas", 3) == 20
    assert expected_items("Inglés", 9) == 22
    assert expected_items("Inglés", 10) == 22
    assert expected_items("Inglés", 11) == 25


def test_performance_level_boundaries():
    assert performance_level(0) == "Nivel 1"
    assert performance_level(25) == "Nivel 1"
    assert performance_level(25.01) == "Nivel 2"
    assert performance_level(50) == "Nivel 2"
    assert performance_level(50.01) == "Nivel 3"
    assert performance_level(75) == "Nivel 3"
    assert performance_level(75.01) == "Nivel 4"
    assert performance_level(100) == "Nivel 4"


def test_grade_and_area_normalization():
    assert grade_number("DECIMO") == 10
    assert grade_number("11°") == 11
    assert normalize_area("MATEMÁTICAS 7°") == "Matemáticas"
    assert normalize_area("SOCIALES Y CIUDADANAS 8°") == "Ciencias sociales"
    assert normalize_area("INGLÉS 10°") == "Inglés"


def test_attempt_score_uses_expected_denominator_and_deduplicates_items():
    raw = pd.DataFrame(
        {
            "AttemptId": [1, 1, 1],
            "Sede": ["MOS", "MOS", "MOS"],
            "Grado": ["TERCERO", "TERCERO", "TERCERO"],
            "IdentiEstudiante": ["A", "A", "A"],
            "Nombre": ["Est", "Est", "Est"],
            "Apellido": ["Uno", "Uno", "Uno"],
            "QuizName": ["MATEMÁTICAS 3°"] * 3,
            "Pregunta": ["P1", "P1", "P2"],
            "IsCorrect": [True, True, True],
        }
    )

    attempts = build_attempt_table(raw)
    row = attempts.iloc[0]

    assert row["Aciertos"] == 2
    assert row["Items_observados"] == 2
    assert row["Items_esperados"] == 20
    assert row["Porcentaje_acierto"] == 10
