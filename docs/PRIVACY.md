# Privacidad y control de divulgación

## Alcance

El proyecto trabaja con resultados educativos de estudiantes, incluyendo
menores de edad. Por tanto, la privacidad se considera un requisito de diseño.

## Datos que nunca deben publicarse

Los agregados del repositorio no deben contener:

- nombre o apellidos;
- correo electrónico;
- documento o identificador de estudiante;
- llaves de emparejamiento;
- filas de nivel estudiante.

## Identificadores indirectos

Una fila puede ser reidentificable incluso sin nombre cuando combina variables
como:

- sede;
- grado;
- curso;
- colegio de origen;
- prueba;
- nivel de desempeño;
- tamaño de grupo muy pequeño.

Por eso la anonimización no se limita a eliminar columnas personales.

## Regla mínima

Las visualizaciones sensibles utilizan como referencia:

```
n >= 5
```

Cuando el tamaño sea menor, el resultado no debe mostrarse de forma que permita
identificar o inferir el resultado de una persona.

En análisis especialmente sensibles puede adoptarse un umbral superior.

## Supresión complementaria

Ocultar una única celda pequeña no siempre es suficiente. Si el total y todas
las demás categorías están visibles, la celda oculta puede deducirse por resta.

Cuando se publiquen tablas completas debe evaluarse la necesidad de suprimir una
segunda celda o usar categorías más agregadas.

## Auditoría

```bash
python scripts/audit_public_data.py
```

El comando reporta:

- errores de lectura;
- columnas con apariencia de identificador directo;
- archivos con celdas pequeñas.

Para convertir las celdas pequeñas en error:

```bash
python scripts/audit_public_data.py --strict-small-cells
```

## Repositorio público

El repositorio es actualmente público.

Eliminar un CSV de la rama principal no elimina automáticamente sus versiones
anteriores del historial Git.

Si los agregados históricos contienen combinaciones potencialmente
reidentificables, existen dos estrategias institucionales:

1. mantener el repositorio privado; o
2. producir una versión pública sanitizada y limpiar explícitamente el historial.

Un cambio de visibilidad o reescritura del historial debe hacerse como decisión
de gobierno, no automáticamente desde el código.

## Historias longitudinales

La Historia longitudinal:

- no publica un expediente individual;
- usa perfiles agregados;
- exige al menos 5 estudiantes por combinación;
- prioriza cohortes con máxima cobertura viable de fuentes.

Debe describirse como un perfil de la **cohorte con máxima cobertura de
fuentes**, no como representación automática de todos los estudiantes del grado.
