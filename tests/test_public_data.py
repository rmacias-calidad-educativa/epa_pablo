from pathlib import Path

import pandas as pd

from data_quality import has_direct_identifier_columns


def test_all_public_csv_files_are_parseable():
    for path in Path("data").glob("*.csv"):
        pd.read_csv(path)


def test_public_csv_files_do_not_expose_direct_identifiers():
    findings = {}
    for path in Path("data").glob("*.csv"):
        df = pd.read_csv(path, nrows=20)
        risky = has_direct_identifier_columns(df)
        if risky:
            findings[path.name] = risky

    assert not findings, findings


def test_story_files_enforce_minimum_group_size():
    story_files = [
        Path("data/student_story_3_9_aggregate_2026.csv"),
        Path("data/student_story_internal_max_sources_2026.csv"),
        Path("data/student_story_latam_10_11_max_sources_2026.csv"),
        Path("data/student_story_profiles_aggregate_2026.csv"),
    ]

    for path in story_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in [
            "n",
            "Estudiantes",
            "Grupo_EFA_n",
            "Candidatos_completos",
        ]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                assert (values >= 5).all(), f"{path.name}: {col}"
