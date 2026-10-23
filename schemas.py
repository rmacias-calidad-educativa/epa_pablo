from __future__ import annotations

SCHEMAS = {
    "latam_detail": {
        "Area", "Sede", "Grado_num", "Categoria_LATAM",
        "Pct_EFA", "Nivel_EFA", "Estudiantes",
    },
    "latam_summary": {
        "Area", "Registros_LATAM", "Emparejados_EFA_misma_area",
        "Cobertura_sobre_LATAM_pct",
    },
    "math_stats": {
        "Sede", "Grado_num", "Metrica", "N_total", "N_validos",
        "Promedio_X", "Promedio_EFA",
    },
    "math_bins": {
        "Sede", "Grado_num", "Metrica", "Rango", "Orden",
        "N_validos", "Promedio_X", "Promedio_EFA",
    },
    "math_summary": {
        "Registros_tabla_nueva", "Emparejados_consistentes",
        "Cobertura_sobre_elegibles_pct",
    },
    "evolution_summary": {
        "Grado_num", "Sede", "Metrica", "N",
        "Sum_Inicial", "Sum_Final", "Sum_Cambio",
    },
    "evolution_trajectory": {
        "Grado_num", "Sede", "Metrica", "Momento",
        "Orden", "N", "Sum_Puntaje",
    },
    "evolution_availability": {
        "Grado_num", "Sede", "Momento", "Registros",
    },
    "evolution_observed": {
        "Grado_num", "Sede", "Momento", "Metrica",
        "N", "Promedio", "Sum_Puntaje",
    },
    "improve_levels": {
        "grado_num", "Sede", "Prueba", "Nivel_EFA",
        "Orden", "n", "Promedio_EFA", "Promedio_Improve",
    },
    "improve_quality": {
        "grado_num", "Sede", "Improve_EV1_estudiantes",
        "Emparejados_identidad", "Cobertura_identidad_pct",
    },
    "improve_growth": {
        "grado_num", "Sede", "Prueba", "Nivel_EFA", "Orden", "n",
        "Promedio_EFA", "Improve_inicial", "Improve_final",
        "Cambio_Improve", "Pct_mejora", "Pct_baja",
    },
    "internal_summary": {
        "Grado_num", "Prueba", "Trimestre", "Nivel_EFA_num",
        "Nivel_EFA", "Estudiantes", "Promedio_interno", "Promedio_EFA",
    },
    "internal_distribution": {
        "Grado_num", "Prueba", "Trimestre", "Nivel_EFA_num",
        "Nivel_EFA", "Interna_nivel", "Estudiantes",
        "Total_grupo", "Porcentaje",
    },
    "internal_coverage": {
        "Grado_num", "Prueba", "EFA_estudiantes",
        "Emparejados", "Cobertura_pct",
    },
    "internal_agreement": {
        "Grado_num", "Prueba", "Trimestre", "Estudiantes",
        "Coincidencia_exacta_pct", "Dentro_un_nivel_pct",
    },
    "internal_reverse": {
        "grado_num", "Prueba", "Trimestre", "nivel_num",
        "Nivel_interno", "Estudiantes", "Promedio_EFA",
    },
    "story_profiles": {
        "grado_num", "Prueba_filtro", "Nivel_EFA_filtro",
        "Candidatos_completos", "Fuentes_maximas", "Fuentes_requeridas",
    },
    "story_3_9": {
        "grado_num", "Prueba_filtro", "Nivel_EFA_filtro",
        "Grupo_EFA_n", "Fuentes_maximas", "Fuentes_requeridas",
        "Fuente", "Resultado", "n", "Valor",
    },
    "story_internal": {
        "grado_num", "Prueba_filtro", "Nivel_EFA_filtro",
        "Grupo_EFA_n", "Fuentes_maximas", "Fuentes_requeridas",
        "Trimestre", "Estudiantes", "Promedio_interno",
    },
    "story_latam_10_11": {
        "grado_num", "Prueba_filtro", "Nivel_EFA_filtro",
        "Grupo_EFA_n", "Fuentes_maximas", "Fuentes_requeridas",
        "Fuente", "Resultado", "n", "Valor",
    },
}
