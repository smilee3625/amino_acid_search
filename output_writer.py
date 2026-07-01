"""Output writing utilities for the PubMed mutation search pipeline.

This module encapsulates writing results to CSV and Excel files using
Pandas. Keeping file I/O separate from data processing simplifies
testing and allows for alternative output formats to be added easily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd  # type: ignore

from . import config


ADDED_RESULT_COLUMNS = [
    "review_paper_count",
    "review_pmid_list",
    "mutation_sentence",
    "effect_sentence",
    "methods_sentence",
    "results_sentence",
    "discussion_sentence",
    "host",
    "cell_line",
    "experiment_type",
    "effect",
    "confidence",
    "effect_terms_found",
    "methods_in_vivo",
    "methods_in_vitro",
    "methods_in_vivo_terms_found",
    "methods_in_vitro_terms_found",
    "evidence_grade",
]


EXCEL_MAX_CELL_CHARS = 32767
EXCEL_SAFE_CELL_CHARS = 32000


def truncate_for_excel(value: Any, max_chars: int = EXCEL_SAFE_CELL_CHARS) -> Any:
    """Truncate long string values so Excel cells do not exceed the limit."""
    if not isinstance(value, str):
        return value
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 20].rstrip() + " ...[excel_truncated]"


def make_excel_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with long text cells truncated for Excel."""
    safe_df = df.copy()
    object_columns = safe_df.select_dtypes(include=["object"]).columns

    for column in object_columns:
        safe_df[column] = safe_df[column].map(truncate_for_excel)

    return safe_df


def build_output_columns(records: List[Dict[str, Any]]) -> List[str]:
    """Build output columns without silently dropping newly added fields."""
    columns: List[str] = []

    for column in config.RESULT_COLUMNS:
        if column not in columns:
            columns.append(column)

    for column in ADDED_RESULT_COLUMNS:
        if column not in columns:
            columns.append(column)

    for record in records:
        for column in record.keys():
            if column not in columns:
                columns.append(column)

    return columns


def save_results(records: Iterable[Dict[str, Any]], csv_path: Path, xlsx_path: Path) -> None:
    """Save results to CSV and Excel files.

    Parameters
    ----------
    records:
        Iterable of dictionaries representing individual mutation search
        results.

    csv_path:
        File path for the CSV output.

    xlsx_path:
        File path for the Excel output.
    """
    record_list = list(records)
    columns = build_output_columns(record_list)

    df = pd.DataFrame(record_list, columns=columns)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    excel_df = make_excel_safe_dataframe(df)
    excel_df.to_excel(xlsx_path, index=False)
