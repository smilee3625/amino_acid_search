"""Query construction for the PubMed mutation search pipeline.

This module builds PubMed queries from influenza segment/mutation pairs.
It expands common segment names and amino-acid mutation formats so that
PubMed searches are less dependent on one exact wording.
"""

from __future__ import annotations

import re

from . import config


SEGMENT_SYNONYMS = {
    "PB2": ["PB2", "polymerase basic protein 2"],
    "PB1": ["PB1", "polymerase basic protein 1"],
    "PA": ["PA", "polymerase acidic protein"],
    "HA": ["HA", "hemagglutinin", "haemagglutinin"],
    "NP": ["NP", "nucleoprotein"],
    "NA": ["NA", "neuraminidase"],
    "M": ["M", "matrix protein"],
    "M1": ["M1", "matrix protein 1"],
    "M2": ["M2", "matrix protein 2"],
    "NS": ["NS", "nonstructural protein"],
    "NS1": ["NS1", "nonstructural protein 1"],
    "NS2": ["NS2", "NEP", "nuclear export protein", "nonstructural protein 2"],
}

AA3 = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
}


def quote_pubmed_phrase(term: str, field: str | None = None) -> str:
    """Quote a PubMed term and optionally add a field tag."""
    term = str(term).strip()
    if not term:
        return ""
    quoted = f'"{term}"'
    if field:
        return f"{quoted}[{field}]"
    return quoted


def build_or_group(terms: list[str], field: str | None = None) -> str:
    """Build a PubMed OR group from unique non-empty terms."""
    cleaned: list[str] = []
    for term in terms:
        term = str(term).strip()
        if term and term not in cleaned:
            cleaned.append(term)

    if not cleaned:
        return ""

    return "(" + " OR ".join(quote_pubmed_phrase(term, field) for term in cleaned) + ")"


def expand_segment_terms(segment: str) -> list[str]:
    """Return common synonyms for an influenza segment/protein name."""
    segment_clean = str(segment).strip()
    key = segment_clean.upper()
    return SEGMENT_SYNONYMS.get(key, [segment_clean])


def expand_mutation_terms(mutation: str) -> list[str]:
    """Return common spelling variants for amino-acid substitutions/deletions."""
    mutation_clean = str(mutation).strip()
    if not mutation_clean:
        return []

    terms = [mutation_clean]
    lower = mutation_clean.lower()

    substitution_match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation_clean.upper())
    if substitution_match:
        ref, pos, alt = substitution_match.groups()
        ref3 = AA3.get(ref, ref)
        alt3 = AA3.get(alt, alt)
        terms.extend([
            f"{ref}{pos}{alt}",
            f"{ref3}{pos}{alt3}",
            f"{ref}{pos} {alt}",
            f"{ref3} {pos} {alt3}",
            f"{ref}{pos} to {alt}",
            f"{ref3}{pos} to {alt3}",
            f"{ref}{pos}{alt} substitution",
            f"{ref3}{pos}{alt3} substitution",
            f"position {pos}",
        ])

    deletion_range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)\s*del", lower)
    if deletion_range_match:
        start, end = deletion_range_match.groups()
        terms.extend([
            f"{start}-{end}del",
            f"{start}-{end} deletion",
            f"deletion {start}-{end}",
            f"deletion at positions {start}-{end}",
            f"positions {start}-{end}",
        ])

    deletion_simple_match = re.fullmatch(r"([a-z]?)(\d+)del", lower)
    if deletion_simple_match:
        aa, pos = deletion_simple_match.groups()
        terms.extend([
            f"{pos}del",
            f"{pos} deletion",
            f"deletion {pos}",
            f"deletion at position {pos}",
        ])
        if aa:
            terms.append(f"{aa.upper()}{pos} deletion")

    return list(dict.fromkeys(terms))


def build_query(segment: str, mutation: str, fulltext_only: bool = False) -> str:
    """Construct an expanded PubMed query for a given segment and mutation."""
    base_query = config.QUERY_TEMPLATE.format(
        segment=segment,
        mutation=mutation,
    )

    segment_group = build_or_group(expand_segment_terms(segment), "Title/Abstract")
    mutation_group = build_or_group(expand_mutation_terms(mutation), "All Fields")

    expanded_parts = [base_query]
    if segment_group:
        expanded_parts.append(segment_group)
    if mutation_group:
        expanded_parts.append(mutation_group)

    query = " AND ".join(f"({part})" for part in expanded_parts if part)

    if fulltext_only:
        query += ' AND "free full text"[Filter]'

    return query
