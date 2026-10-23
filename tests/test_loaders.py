from loaders import (
    read_default_data,
    read_evolution_10_11_data,
    read_improve_efa_data,
    read_internal_efa_data,
    read_latam_efa_data,
    read_math_activity_data,
    read_student_story_3_9,
    read_student_story_profiles,
    read_student_story_support,
)


def test_default_bundle_loads():
    frames = read_default_data()
    assert len(frames) == 8
    assert not frames[0].empty


def test_latam_loader_contract():
    detail, summary = read_latam_efa_data()
    assert {"Area", "Sede", "Grado_num", "Estudiantes"}.issubset(
        detail.columns
    )
    assert not summary.empty


def test_math_activity_loader_contract():
    stats, bins, summary = read_math_activity_data()
    assert {"Sede", "Grado_num", "Metrica"}.issubset(stats.columns)
    assert {"Sede", "Grado_num", "Metrica", "N_validos"}.issubset(
        bins.columns
    )
    assert not summary.empty


def test_evolution_loader_contract():
    summary, trajectory, availability, observed = (
        read_evolution_10_11_data()
    )
    assert not summary.empty
    assert not trajectory.empty
    assert not availability.empty
    assert not observed.empty


def test_improve_efa_loader_contract():
    levels, quality, growth = read_improve_efa_data()
    assert {"grado_num", "Prueba", "Nivel_EFA", "n"}.issubset(
        levels.columns
    )
    assert not quality.empty
    assert {"grado_num", "Prueba", "Nivel_EFA", "n"}.issubset(
        growth.columns
    )


def test_internal_efa_loader_contract():
    summary, distribution, coverage, agreement, reverse = (
        read_internal_efa_data()
    )
    assert {"Grado_num", "Prueba", "Trimestre"}.issubset(
        summary.columns
    )
    assert {"Grado_num", "Prueba", "Estudiantes"}.issubset(
        distribution.columns
    )
    assert not coverage.empty
    assert not agreement.empty
    assert not reverse.empty


def test_story_loader_contracts():
    profiles = read_student_story_profiles()
    profiles_3_9 = read_student_story_3_9()
    internal, latam = read_student_story_support()

    assert {
        "grado_num",
        "Prueba_filtro",
        "Nivel_EFA_filtro",
        "Fuentes_maximas",
        "Fuentes_requeridas",
    }.issubset(profiles.columns)

    assert {
        "grado_num",
        "Prueba_filtro",
        "Nivel_EFA_filtro",
        "Fuentes_maximas",
        "Fuentes_requeridas",
    }.issubset(profiles_3_9.columns)

    assert not internal.empty
    assert not latam.empty
