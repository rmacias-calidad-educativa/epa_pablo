from __future__ import annotations

import base64
import gzip
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    ANTIGUEDAD_SITES,
    ANTIGUEDAD_FULL_SITES,
    LEGACY_LEVEL_MAP,
    MIN_PUBLIC_GROUP_SIZE,
    SITES,
    TESTS,
    YEAR,
)
from data_quality import read_csv_validated, suppress_small_groups
from processing import (
    build_attempt_table,
    build_dimension_table,
    normalize_area,
)
from schemas import SCHEMAS


@st.cache_data(show_spinner=False)
def read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_latam_efa_data():
    detail_path = Path("data/latam_efa_aggregate_2026.csv")
    summary_path = Path("data/latam_efa_match_summary_2026.csv")

    detail = read_csv_validated(
        detail_path,
        SCHEMAS["latam_detail"],
    )
    summary = read_csv_validated(
        summary_path,
        SCHEMAS["latam_summary"],
    )

    if not detail.empty:
        detail = detail[
            detail["Sede"].astype(str).isin(SITES)
        ].copy()
        detail["Grado_num"] = pd.to_numeric(
            detail["Grado_num"], errors="coerce"
        ).astype("Int64")
        detail["Pct_EFA"] = pd.to_numeric(
            detail["Pct_EFA"], errors="coerce"
        )
        detail["Estudiantes"] = pd.to_numeric(
            detail["Estudiantes"], errors="coerce"
        ).fillna(0).astype(int)
        detail = suppress_small_groups(
            detail,
            "Estudiantes",
            MIN_PUBLIC_GROUP_SIZE,
        )

    return detail, summary


@st.cache_data(show_spinner=False)
def read_math_activity_data():
    stats_path = Path("data/math_activity_stats_2026.csv")
    bins_path = Path("data/math_activity_bins_2026.csv")
    summary_path = Path("data/math_activity_match_summary_2026.csv")

    stats = read_csv_validated(
        stats_path,
        SCHEMAS["math_stats"],
    )
    bins = read_csv_validated(
        bins_path,
        SCHEMAS["math_bins"],
    )
    summary = read_csv_validated(
        summary_path,
        SCHEMAS["math_summary"],
    )

    for df in [stats, bins]:
        if not df.empty:
            df["Grado_num"] = pd.to_numeric(
                df["Grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["Grado_num"], inplace=True)
            df["Grado_num"] = df["Grado_num"].astype(int)
            df.drop(
                df[~df["Sede"].astype(str).isin(SITES)].index,
                inplace=True,
            )

    return stats, bins, summary


@st.cache_data(show_spinner=False)
def read_evolution_10_11_data():
    summary_path = Path("data/evolucion_10_11_summary_2026.csv")
    trajectory_path = Path("data/evolucion_11_trayectoria_2026.csv")
    availability_path = Path(
        "data/evolucion_10_11_disponibilidad_2026.csv"
    )
    observed_path = Path(
        "data/evolucion_10_11_observado_2026.csv"
    )

    summary = read_csv_validated(
        summary_path,
        SCHEMAS["evolution_summary"],
    )
    trajectory = read_csv_validated(
        trajectory_path,
        SCHEMAS["evolution_trajectory"],
    )
    availability = read_csv_validated(
        availability_path,
        SCHEMAS["evolution_availability"],
    )
    observed = read_csv_validated(
        observed_path,
        SCHEMAS["evolution_observed"],
    )

    for df in [summary, trajectory, availability, observed]:
        if not df.empty and "Grado_num" in df.columns:
            df["Grado_num"] = pd.to_numeric(
                df["Grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["Grado_num"], inplace=True)
            df["Grado_num"] = df["Grado_num"].astype(int)

    return summary, trajectory, availability, observed


@st.cache_data(show_spinner=False)
def read_improve_efa_data():
    levels_path = Path("data/improve_efa_levels_2026.csv")
    quality_path = Path("data/improve_efa_match_quality_2026.csv")
    growth_path = Path("data/improve_growth_by_efa_level_2026.csv")

    levels = read_csv_validated(
        levels_path,
        SCHEMAS["improve_levels"],
    )
    quality = read_csv_validated(
        quality_path,
        SCHEMAS["improve_quality"],
    )
    growth = read_csv_validated(
        growth_path,
        SCHEMAS["improve_growth"],
    )

    for df in [levels, quality, growth]:
        if not df.empty and "grado_num" in df.columns:
            df["grado_num"] = pd.to_numeric(
                df["grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["grado_num"], inplace=True)
            df["grado_num"] = df["grado_num"].astype(int)

    if not levels.empty:
        levels["Orden"] = pd.to_numeric(
            levels["Orden"], errors="coerce"
        )
        levels["n"] = pd.to_numeric(
            levels["n"], errors="coerce"
        ).fillna(0).astype(int)
        levels["Promedio_EFA"] = pd.to_numeric(
            levels["Promedio_EFA"], errors="coerce"
        )
        levels["Promedio_Improve"] = pd.to_numeric(
            levels["Promedio_Improve"], errors="coerce"
        )

    if not growth.empty:
        for col in [
            "Orden",
            "n",
            "Promedio_EFA",
            "Improve_inicial",
            "Improve_final",
            "Cambio_Improve",
            "Pct_mejora",
            "Pct_baja",
        ]:
            growth[col] = pd.to_numeric(
                growth[col], errors="coerce"
            )
        growth["n"] = growth["n"].fillna(0).astype(int)

    levels = suppress_small_groups(
        levels,
        "n",
        MIN_PUBLIC_GROUP_SIZE,
    )
    growth = suppress_small_groups(
        growth,
        "n",
        MIN_PUBLIC_GROUP_SIZE,
    )

    return levels, quality, growth


@st.cache_data(show_spinner=False)
def read_internal_efa_data():
    summary_path = Path("data/internal_efa_summary_2026.csv")
    distribution_path = Path("data/internal_efa_distribution_2026.csv")
    coverage_path = Path("data/internal_efa_coverage_2026.csv")
    agreement_path = Path("data/internal_efa_agreement_2026.csv")
    reverse_path = Path("data/internal_efa_reverse_t1_t2_2026.csv")

    summary = read_csv_validated(
        summary_path,
        SCHEMAS["internal_summary"],
    )
    distribution = read_csv_validated(
        distribution_path,
        SCHEMAS["internal_distribution"],
    )
    coverage = read_csv_validated(
        coverage_path,
        SCHEMAS["internal_coverage"],
    )
    agreement = read_csv_validated(
        agreement_path,
        SCHEMAS["internal_agreement"],
    )
    reverse = read_csv_validated(
        reverse_path,
        SCHEMAS["internal_reverse"],
    )

    for df in [summary, distribution, coverage, agreement, reverse]:
        if not df.empty and "Grado_num" in df.columns:
            df["Grado_num"] = pd.to_numeric(
                df["Grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["Grado_num"], inplace=True)
            df["Grado_num"] = df["Grado_num"].astype(int)

    if not summary.empty:
        summary["Nivel_EFA_num"] = pd.to_numeric(
            summary["Nivel_EFA_num"], errors="coerce"
        )
        summary["Estudiantes"] = pd.to_numeric(
            summary["Estudiantes"], errors="coerce"
        ).fillna(0).astype(int)
        summary["Promedio_interno"] = pd.to_numeric(
            summary["Promedio_interno"], errors="coerce"
        )
        summary["Promedio_EFA"] = pd.to_numeric(
            summary["Promedio_EFA"], errors="coerce"
        )

    if not distribution.empty:
        distribution["Nivel_EFA_num"] = pd.to_numeric(
            distribution["Nivel_EFA_num"], errors="coerce"
        )
        distribution["Estudiantes"] = pd.to_numeric(
            distribution["Estudiantes"], errors="coerce"
        ).fillna(0).astype(int)
        distribution["Porcentaje"] = pd.to_numeric(
            distribution["Porcentaje"], errors="coerce"
        )

    if not coverage.empty:
        for col in ["EFA_estudiantes", "Emparejados"]:
            coverage[col] = pd.to_numeric(
                coverage[col], errors="coerce"
            ).fillna(0).astype(int)
        coverage["Cobertura_pct"] = pd.to_numeric(
            coverage["Cobertura_pct"], errors="coerce"
        )

    if not agreement.empty:
        for col in [
            "Coincidencia_exacta_pct",
            "Dentro_un_nivel_pct",
            "Promedio_interno",
            "Promedio_nivel_EFA",
        ]:
            agreement[col] = pd.to_numeric(
                agreement[col], errors="coerce"
            )

    if not reverse.empty:
        for col in [
            "grado_num",
            "nivel_num",
            "Estudiantes",
            "Promedio_EFA",
            "Mediana_EFA",
            "DE_EFA",
            "Trimestre_orden",
        ]:
            reverse[col] = pd.to_numeric(
                reverse[col], errors="coerce"
            )
        reverse["grado_num"] = reverse[
            "grado_num"
        ].astype("Int64")
        reverse.dropna(subset=["grado_num"], inplace=True)
        reverse["grado_num"] = reverse["grado_num"].astype(int)
        reverse["Estudiantes"] = reverse[
            "Estudiantes"
        ].fillna(0).astype(int)

    return summary, distribution, coverage, agreement, reverse


@st.cache_data(show_spinner=False)
def read_student_story_profiles():
    # Cache schema v2: perfiles con máxima cobertura de fuentes.
    path = Path("data/student_story_profiles_aggregate_2026.csv")
    data = read_csv_validated(
        path,
        SCHEMAS["story_profiles"],
    )

    if not data.empty:

        data["grado_num"] = pd.to_numeric(
            data["grado_num"], errors="coerce"
        ).astype("Int64")
        data.dropna(subset=["grado_num"], inplace=True)
        data["grado_num"] = data["grado_num"].astype(int)
        data["Candidatos_completos"] = pd.to_numeric(
            data["Candidatos_completos"], errors="coerce"
        ).fillna(0).astype(int)
        data["Fuentes_maximas"] = pd.to_numeric(
            data["Fuentes_maximas"], errors="coerce"
        ).fillna(0).astype(int)

        for col in data.columns:
            if (
                col.startswith("EFA_")
                or col.startswith("Improve_")
            ) and col not in [
                "Prueba_filtro",
                "Nivel_EFA_filtro",
            ]:
                data[col] = pd.to_numeric(
                    data[col], errors="coerce"
                )

    return data


@st.cache_data(show_spinner=False)
def read_student_story_3_9():
    # Cache schema v2: perfiles con máxima cobertura de fuentes.
    path = Path("data/student_story_3_9_aggregate_2026.csv")
    data = read_csv_validated(
        path,
        SCHEMAS["story_3_9"],
    )

    if not data.empty:

        data["grado_num"] = pd.to_numeric(
            data["grado_num"], errors="coerce"
        ).astype("Int64")
        data.dropna(subset=["grado_num"], inplace=True)
        data["grado_num"] = data["grado_num"].astype(int)
        data["Grupo_EFA_n"] = pd.to_numeric(
            data["Grupo_EFA_n"], errors="coerce"
        ).fillna(0).astype(int)
        data["Fuentes_maximas"] = pd.to_numeric(
            data["Fuentes_maximas"], errors="coerce"
        ).fillna(0).astype(int)
        data["n"] = pd.to_numeric(
            data["n"], errors="coerce"
        ).fillna(0).astype(int)
        data["Valor"] = pd.to_numeric(
            data["Valor"], errors="coerce"
        )
        data["Escala_max"] = pd.to_numeric(
            data["Escala_max"], errors="coerce"
        )
        data["Extra"] = pd.to_numeric(
            data["Extra"], errors="coerce"
        )

    return data


@st.cache_data(show_spinner=False)
def read_student_story_support():
    internal_path = Path(
        "data/student_story_internal_max_sources_2026.csv"
    )
    latam_path = Path(
        "data/student_story_latam_10_11_max_sources_2026.csv"
    )

    internal = read_csv_validated(
        internal_path,
        SCHEMAS["story_internal"],
    )
    latam = read_csv_validated(
        latam_path,
        SCHEMAS["story_latam_10_11"],
    )

    for df in [internal, latam]:
        if not df.empty:
            df["grado_num"] = pd.to_numeric(
                df["grado_num"], errors="coerce"
            ).astype("Int64")
            df.dropna(subset=["grado_num"], inplace=True)
            df["grado_num"] = df["grado_num"].astype(int)

    if not internal.empty:
        internal["Estudiantes"] = pd.to_numeric(
            internal["Estudiantes"], errors="coerce"
        ).fillna(0).astype(int)
        internal["Promedio_interno"] = pd.to_numeric(
            internal["Promedio_interno"], errors="coerce"
        )

    if not latam.empty:
        latam["n"] = pd.to_numeric(
            latam["n"], errors="coerce"
        ).fillna(0).astype(int)
        latam["Valor"] = pd.to_numeric(
            latam["Valor"], errors="coerce"
        )
        latam["Extra"] = pd.to_numeric(
            latam["Extra"], errors="coerce"
        )

    return internal, latam


def _age_scope_mask(frame: pd.DataFrame) -> pd.Series:
    """Sedes nuevas completas; en las demás, antigüedad individual <= 1."""
    site = frame["Sede"].astype(str)
    years = pd.to_numeric(frame["AntiguedadBS"], errors="coerce")
    return site.isin(ANTIGUEDAD_FULL_SITES) | (
        site.isin(ANTIGUEDAD_SITES) & years.le(1)
    )


def _include_full_new_sites(
    temporal: pd.DataFrame,
    levels: pd.DataFrame,
    dimensions: pd.DataFrame,
    origin: pd.DataFrame,
    age_temporal: pd.DataFrame,
    age_levels: pd.DataFrame,
    age_dimensions: pd.DataFrame,
    age_origin: pd.DataFrame,
):
    """Reutiliza agregados generales para BAQ/USAQ sin duplicar sedes."""
    general_frames = (temporal, levels, dimensions, origin)
    age_frames = (age_temporal, age_levels, age_dimensions, age_origin)
    revised = []
    for general, age in zip(general_frames, age_frames):
        if general is None or general.empty or "Sede" not in general.columns:
            revised.append(age)
            continue
        full = general[
            general["Sede"].astype(str).isin(ANTIGUEDAD_FULL_SITES)
        ].copy()
        if age is None or age.empty:
            revised.append(full)
            continue
        remaining = age[
            ~age["Sede"].astype(str).isin(ANTIGUEDAD_FULL_SITES)
        ].copy()
        revised.append(pd.concat([remaining, full], ignore_index=True))
    return tuple(revised)


@st.cache_data(show_spinner=False)
def process_raw(raw: pd.DataFrame):
    attempts = build_attempt_table(raw)
    attempts["Prueba"] = attempts["QuizName"].apply(normalize_area)
    attempts = attempts[
        (attempts["Año"] == YEAR)
        & (attempts["Prueba"].isin(TESTS))
        & (attempts["Sede"].astype(str).isin(SITES))
    ].copy()

    dimensions = build_dimension_table(raw)
    if not dimensions.empty:
        dimensions = dimensions[
            (dimensions["Año"] == YEAR)
            & (dimensions["Prueba"].isin(TESTS))
            & (dimensions["Sede"].astype(str).isin(SITES))
        ].copy()

    age_attempts = attempts[_age_scope_mask(attempts)].copy()

    if not dimensions.empty and "AntiguedadBS" in dimensions.columns:
        age_dimensions = dimensions[_age_scope_mask(dimensions)].copy()
    else:
        raw_age = raw[_age_scope_mask(raw)].copy()
        age_dimensions = build_dimension_table(raw_age)
        if not age_dimensions.empty:
            age_dimensions = age_dimensions[
                (age_dimensions["Año"] == YEAR)
                & (age_dimensions["Prueba"].isin(TESTS))
            ].copy()

    return attempts, dimensions, age_attempts, age_dimensions


@st.cache_data(show_spinner=False)
def read_default_data():
    bundle_paths = [
        Path(f"data/dashboard_bundle_20261019_part{i:02d}.b64")
        for i in range(14)
    ]
    if all(path.exists() for path in bundle_paths):
        payload = "".join(
            path.read_text(encoding="utf-8").strip()
            for path in bundle_paths
        )
        bundle = json.loads(
            gzip.decompress(
                base64.b64decode(payload)
            ).decode("utf-8")
        )

        temporal = pd.read_csv(
            io.StringIO(bundle["general_temporal"])
        )
        levels = pd.read_csv(
            io.StringIO(bundle["general_levels"])
        )
        dimensions = pd.read_csv(
            io.StringIO(bundle["general_dimensions"])
        )
        top10_origin = pd.read_csv(
            io.StringIO(bundle["general_top10"])
        )
        age_temporal = pd.read_csv(
            io.StringIO(bundle["age_temporal"])
        )
        age_levels = pd.read_csv(
            io.StringIO(bundle["age_levels"])
        )
        age_dimensions = pd.read_csv(
            io.StringIO(bundle["age_dimensions"])
        )
        age_top10 = pd.read_csv(
            io.StringIO(bundle["age_top10"])
        )

        if "Prueba" not in temporal.columns:
            temporal["Prueba"] = temporal["QuizName"].apply(
                normalize_area
            )

        for df in [
            levels,
            top10_origin,
            age_levels,
            age_top10,
        ]:
            if not df.empty and "Nivel" in df.columns:
                df["Nivel"] = df["Nivel"].replace(
                    LEGACY_LEVEL_MAP
                )

        if (
            not dimensions.empty
            and "competencia" in dimensions.columns
        ):
            dimensions = dimensions.rename(
                columns={"competencia": "Dimension"}
            )
        if (
            not age_dimensions.empty
            and "competencia" in age_dimensions.columns
        ):
            age_dimensions = age_dimensions.rename(
                columns={"competencia": "Dimension"}
            )

        if "Año" in temporal.columns:
            temporal = temporal[
                temporal["Año"] == YEAR
            ].copy()
        if "Año" in age_temporal.columns:
            age_temporal = age_temporal[
                age_temporal["Año"] == YEAR
            ].copy()

        temporal = temporal[
            temporal["Prueba"].isin(TESTS)
        ].copy()
        age_temporal = age_temporal[
            age_temporal["Prueba"].isin(TESTS)
        ].copy()

        # NIZA se excluye completamente del análisis general.
        temporal = temporal[
            temporal["Sede"].astype(str).isin(SITES)
        ].copy()
        if not levels.empty and "Sede" in levels.columns:
            levels = levels[
                levels["Sede"].astype(str).isin(SITES)
            ].copy()
        if not dimensions.empty and "Sede" in dimensions.columns:
            dimensions = dimensions[
                dimensions["Sede"].astype(str).isin(SITES)
            ].copy()
        if not top10_origin.empty and "Sede" in top10_origin.columns:
            top10_origin = top10_origin[
                top10_origin["Sede"].astype(str).isin(SITES)
            ].copy()

        return _include_full_new_sites(
            temporal,
            levels,
            dimensions,
            top10_origin,
            age_temporal,
            age_levels,
            age_dimensions,
            age_top10,
        )

    temporal_path = Path("data/default_temporal_summary.csv")
    if not temporal_path.exists():
        return tuple(pd.DataFrame() for _ in range(8))

    temporal = pd.read_csv(temporal_path)
    temporal["Prueba"] = temporal["QuizName"].apply(normalize_area)
    if "Año" in temporal.columns:
        temporal = temporal[temporal["Año"] == YEAR]
    temporal = temporal[temporal["Prueba"].isin(TESTS)].copy()

    level_files = sorted(Path("data").glob("default_levels_2026_part*.csv"))
    levels = (
        pd.concat(
            [pd.read_csv(path) for path in level_files],
            ignore_index=True,
        )
        if level_files
        else pd.DataFrame()
    )
    if not levels.empty and "Nivel" in levels.columns:
        levels["Nivel"] = levels["Nivel"].replace(LEGACY_LEVEL_MAP)

    dimension_files = sorted(
        Path("data").glob("default_dimensions_2026_part*.csv")
    )
    dimensions = (
        pd.concat(
            [pd.read_csv(path) for path in dimension_files],
            ignore_index=True,
        )
        if dimension_files
        else pd.DataFrame()
    )
    if not dimensions.empty and "competencia" in dimensions.columns:
        dimensions = dimensions.rename(
            columns={"competencia": "Dimension"}
        )

    top10_grade_files = sorted(
        Path("data").glob("top10_origin_by_site_grade_*.csv")
    )
    top10_origin = (
        pd.concat(
            [pd.read_csv(path) for path in top10_grade_files],
            ignore_index=True,
        )
        if top10_grade_files
        else pd.DataFrame()
    )
    if not top10_origin.empty and "Nivel" in top10_origin.columns:
        top10_origin["Nivel"] = top10_origin["Nivel"].replace(
            LEGACY_LEVEL_MAP
        )

    age_temporal_path = Path("data/antiguedad_temporal_2026.csv")
    age_temporal = (
        pd.read_csv(age_temporal_path)
        if age_temporal_path.exists()
        else pd.DataFrame()
    )
    if not age_temporal.empty:
        age_temporal = age_temporal[
            age_temporal["Prueba"].isin(TESTS)
        ].copy()

    age_levels_path = Path("data/antiguedad_levels_2026.csv")
    age_levels = (
        pd.read_csv(age_levels_path)
        if age_levels_path.exists()
        else pd.DataFrame()
    )
    if not age_levels.empty and "Nivel" in age_levels.columns:
        age_levels["Nivel"] = age_levels["Nivel"].replace(
            LEGACY_LEVEL_MAP
        )

    age_dimensions_path = Path("data/antiguedad_dimensions_2026.csv")
    age_dimensions = (
        pd.read_csv(age_dimensions_path)
        if age_dimensions_path.exists()
        else pd.DataFrame()
    )

    age_top10_path = Path("data/antiguedad_top10_2026.csv")
    age_top10 = (
        pd.read_csv(age_top10_path)
        if age_top10_path.exists()
        else pd.DataFrame()
    )
    if not age_top10.empty and "Nivel" in age_top10.columns:
        age_top10["Nivel"] = age_top10["Nivel"].replace(
            LEGACY_LEVEL_MAP
        )

    # NIZA se excluye completamente del análisis general.
    temporal = temporal[
        temporal["Sede"].astype(str).isin(SITES)
    ].copy()
    if not levels.empty and "Sede" in levels.columns:
        levels = levels[
            levels["Sede"].astype(str).isin(SITES)
        ].copy()
    if not dimensions.empty and "Sede" in dimensions.columns:
        dimensions = dimensions[
            dimensions["Sede"].astype(str).isin(SITES)
        ].copy()
    if not top10_origin.empty and "Sede" in top10_origin.columns:
        top10_origin = top10_origin[
            top10_origin["Sede"].astype(str).isin(SITES)
        ].copy()

    return _include_full_new_sites(
        temporal,
        levels,
        dimensions,
        top10_origin,
        age_temporal,
        age_levels,
        age_dimensions,
        age_top10,
    )
