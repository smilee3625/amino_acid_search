"""Input parsing utilities for the PubMed mutation search pipeline.

This module reads user-provided segment/mutation input files. It accepts
multiple simple formats, including tab-separated, comma-separated,
space-separated, and compact delimiter forms such as ``PB2:D701N``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Tuple


HEADER_SEGMENT_TERMS = {
    "segment",
    "protein",
    "gene",
    "genesegment",
    "gene_segment",
}

HEADER_MUTATION_TERMS = {
    "mutation",
    "mutation/motif",
    "motif",
    "aa_change",
    "aachange",
    "amino_acid_change",
}

KNOWN_SEGMENTS = {
    "PB2",
    "PB1",
    "PA",
    "HA",
    "NP",
    "NA",
    "M",
    "M1",
    "M2",
    "NS",
    "NS1",
    "NS2",
}


def normalize_header_token(token: str) -> str:
    """Normalize a possible header token for comparison."""
    return re.sub(r"\s+", "", str(token).strip().lower())


def is_header_row(columns: list[str]) -> bool:
    """Return True if a row looks like a header row."""
    if len(columns) < 2:
        return False

    first = normalize_header_token(columns[0])
    second = normalize_header_token(columns[1])

    return first in HEADER_SEGMENT_TERMS and second in HEADER_MUTATION_TERMS


def clean_value(value: str) -> str:
    """Clean one parsed cell value."""
    return str(value).strip().strip('"').strip("'").strip()


def split_line_flexibly(line: str) -> tuple[str, str] | None:
    """Parse one line into segment and mutation using multiple simple formats."""
    line = line.strip()
    if not line:
        return None

    if line.startswith("#"):
        return None

    # First try CSV/TSV parsing. This handles quoted fields safely.
    for delimiter in ["\t", ","]:
        parsed = next(csv.reader([line], delimiter=delimiter))
        columns = [clean_value(col) for col in parsed if clean_value(col)]
        if is_header_row(columns):
            return None
        if len(columns) >= 2:
            return columns[0], columns[1]

    # Then try compact delimiter forms: PB2:D701N, PB2/D701N, PB2|D701N.
    compact_match = re.match(r"^([A-Za-z0-9]+)\s*[:/|]\s*(.+)$", line)
    if compact_match:
        segment, mutation = compact_match.groups()
        return clean_value(segment), clean_value(mutation)

    # Handle hyphen only when the left side is a known segment to avoid
    # breaking mutation strings such as 80-84del.
    hyphen_match = re.match(r"^([A-Za-z0-9]+)\s+-\s+(.+)$", line)
    if hyphen_match:
        segment, mutation = hyphen_match.groups()
        if segment.upper() in KNOWN_SEGMENTS:
            return clean_value(segment), clean_value(mutation)

    # Finally try whitespace-separated rows: PB2 D701N.
    parts = line.split()
    if len(parts) >= 2:
        if is_header_row(parts):
            return None
        return clean_value(parts[0]), clean_value(" ".join(parts[1:]))

    return None


def parse_input_rows(file_path: Path) -> list[Tuple[str, str]]:
    """Parse amino acid change rows from a user input file.

    Accepted examples
    -----------------
    - ``PB2\tD701N``
    - ``PB2,D701N``
    - ``PB2 D701N``
    - ``PB2:D701N``
    - ``PB2/D701N``
    - ``# comment`` is ignored

    Returns
    -------
    list of (segment, mutation):
        Each tuple contains the segment and mutation/motif.
    """
    rows: list[Tuple[str, str]] = []

    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        for line_number, line in enumerate(f, start=1):
            parsed = split_line_flexibly(line)
            if parsed is None:
                continue

            segment, mutation = parsed
            if not segment or not mutation:
                raise ValueError(f"Invalid segment/mutation at line {line_number}: {line.rstrip()}")

            rows.append((segment, mutation))

    if not rows:
        raise ValueError(f"No valid segment/mutation rows found in {file_path}")

    return rows
