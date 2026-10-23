from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DIRECT_IDENTIFIER_TOKENS = {
    "correo",
    "email",
    "nombre",
    "apellido",
    "documento",
    "identiestudiante",
    "identificacion",
    "identificación",
}


def require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    source: str = "dataset",
) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(
            f"{source}: faltan columnas requeridas: {', '.join(missing)}"
        )


def suppress_small_groups(
    df: pd.DataFrame,
    count_col: str,
    min_n: int = 5,
) -> pd.DataFrame:
    """Excluye grupos pequeños antes de mostrarlos públicamente."""
    if df.empty or count_col not in df.columns:
        return df.copy()

    out = df.copy()
    counts = pd.to_numeric(out[count_col], errors="coerce")
    return out[counts >= min_n].copy()


def has_direct_identifier_columns(df: pd.DataFrame) -> list[str]:
    risky = []
    for col in df.columns:
        normalized = (
            str(col)
            .strip()
            .lower()
            .replace("_", "")
            .replace(" ", "")
        )
        if any(token.replace("_", "") in normalized for token in DIRECT_IDENTIFIER_TOKENS):
            risky.append(str(col))
    return risky


def read_csv_validated(
    path: str | Path,
    required: Iterable[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"No fue posible leer {path.as_posix()}: {exc}"
        ) from exc

    if required:
        require_columns(df, required, path.as_posix())

    return df
