from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_quality import has_direct_identifier_columns


COUNT_COLUMNS = [
    "n",
    "N",
    "Estudiantes",
    "N_validos",
    "Pares_validos",
    "Emparejados",
    "Grupo_EFA_n",
    "Candidatos_completos",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-small-cells",
        action="store_true",
        help="Falla si encuentra grupos con n < 5.",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    csv_files = sorted(data_dir.glob("*.csv"))

    parse_errors: list[str] = []
    direct_identifier_files: list[str] = []
    small_cell_findings: list[str] = []

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            parse_errors.append(f"{path}: {exc}")
            continue

        risky_cols = has_direct_identifier_columns(df)
        if risky_cols:
            direct_identifier_files.append(
                f"{path}: {', '.join(risky_cols)}"
            )

        for col in COUNT_COLUMNS:
            if col not in df.columns:
                continue
            counts = pd.to_numeric(df[col], errors="coerce")
            small = int((counts < 5).sum())
            if small:
                minimum = counts.min()
                small_cell_findings.append(
                    f"{path.name}: {small} filas con {col}<5 "
                    f"(mínimo={minimum:g})"
                )

    if parse_errors:
        print("\nCSV con errores de lectura:")
        for item in parse_errors:
            print(" -", item)

    if direct_identifier_files:
        print("\nArchivos con columnas que parecen identificadores directos:")
        for item in direct_identifier_files:
            print(" -", item)

    if small_cell_findings:
        print("\nCeldas pequeñas detectadas:")
        for item in small_cell_findings:
            print(" -", item)

    if parse_errors or direct_identifier_files:
        return 1

    if args.strict_small_cells and small_cell_findings:
        return 2

    print(
        f"\nAuditoría completada: {len(csv_files)} CSV legibles, "
        "sin identificadores directos detectados."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
