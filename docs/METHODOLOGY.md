# Metodología analítica

## Propósito

El dashboard integra distintas fuentes de evaluación para describir el desempeño
y las trayectorias de aprendizaje. Su objetivo es apoyar la toma de decisiones
académicas, no producir diagnósticos causales ni clasificaciones psicométricas
oficiales.

## Evaluar para Avanzar

### Puntaje

El puntaje se calcula como:

```
aciertos / ítems esperados × 100
```

Los ítems no observados no reducen el denominador esperado. Esto evita que una
aplicación incompleta infle artificialmente el resultado.

Ítems esperados:

- Matemáticas, Lenguaje, Ciencias naturales y Ciencias sociales: 20.
- Inglés 9° y 10°: 22.
- Inglés 11°: 25.

### Bandas analíticas

- Nivel 1: 0–25%.
- Nivel 2: >25–50%.
- Nivel 3: >50–75%.
- Nivel 4: >75–100%.

Estas bandas son una taxonomía analítica interna. **No son niveles oficiales de
desempeño definidos mediante un procedimiento de standard setting.**

## Emparejamiento de fuentes

Siempre que sea posible, los cruces se realizan a nivel del mismo estudiante.

- Evaluación interna ↔ EFA: correo exacto normalizado, con validación de grado y sede.
- Improve ↔ EFA: documento validado + mismo grado + misma sede.
- LATAM ↔ EFA: correo mediante la base puente de usuarios y luego identificador EFA.
- Innovamat ↔ EFA: identificador disponible en las bases de integración.

No se utilizan coincidencias difusas por nombre para los análisis publicados.

## Evaluación interna

La escala interna se interpreta ordinalmente:

1. Progreso limitado
2. Emergente
3. En aceleración
4. Avanzado

La comparación con EFA representa **convergencia de evidencias**, no equivalencia
entre instrumentos.

En la vista temporal se utilizan Trimestre 1 y Trimestre 2.

## Improve

Los análisis de crecimiento utilizan los mismos estudiantes entre aplicaciones:

- 10°: Evaluación 1 → Evaluación 2.
- 11°: Evaluación 1 → Evaluación 3.

Los promedios observados de aplicaciones completas no deben interpretarse como
crecimiento si las poblaciones no son idénticas.

## Historia longitudinal

La historia longitudinal usa perfiles agregados, no expedientes individuales.

Solo se habilitan combinaciones con al menos 5 estudiantes y se prioriza la
máxima cantidad viable de fuentes por grado.

Por tanto, el perfil debe interpretarse como:

> perfil representativo de estudiantes con máxima cobertura de fuentes

y no como perfil de todos los estudiantes del grado.

## Privacidad

El repositorio no debe contener nombres, correos, documentos ni identificadores
individuales en los agregados públicos.

Las vistas sensibles aplican una regla mínima de divulgación de **n ≥ 5**.
El script `scripts/audit_public_data.py` audita archivos públicos y reporta
celdas pequeñas.

Importante: si el repositorio permanece público, eliminar un archivo en el
estado actual no elimina automáticamente versiones anteriores del historial Git.

## Interpretación

Los resultados del dashboard son principalmente:

- descriptivos;
- longitudinales cuando la muestra es emparejada;
- comparativos;
- de convergencia entre fuentes.

No deben utilizarse por sí solos para inferir causalidad, evaluar docentes o
tomar decisiones individuales de alto impacto.
