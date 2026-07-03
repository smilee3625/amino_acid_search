"""Query construction for the PubMed mutation search pipeline.

This module builds PubMed queries from influenza segment/mutation pairs.
It expands common segment names and amino-acid mutation formats so that
PubMed searches are less dependent on one exact wording.
"""

from __future__ import annotations

import re


SEGMENT_SYNONYMS = {
    "PB2": ["PB2", "polymerase basic protein 2", "polymerase basic 2"],
    "PB1": ["PB1", "polymerase basic protein 1", "polymerase basic 1"],
    "PA": ["PA", "polymerase acidic protein", "polymerase acidic"],
    "HA": ["HA", "hemagglutinin", "haemagglutinin"],
    "HA2": ["HA2", "hemagglutinin", "haemagglutinin"],
    "NP": ["NP", "nucleoprotein"],
    "NA": ["NA", "neuraminidase"],
    "NA(N1)": ["NA", "N1", "neuraminidase"],
    "NA(N6)": ["NA", "N6", "neuraminidase"],
    "M": ["M", "matrix protein"],
    "M1": ["M1", "matrix protein 1"],
    "M2": ["M2", "matrix protein 2", "ion channel"],
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


def add_unique(terms: list[str], new_terms: list[str]) -> None:
    """Append non-empty terms while preserving order and removing duplicates."""
    for term in new_terms:
        cleaned = str(term).strip()
        if cleaned and cleaned not in terms:
            terms.append(cleaned)


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
    """Return broad spelling variants for mutation, residue, motif, and deletion markers."""
    mutation_clean = str(mutation).strip()
    if not mutation_clean:
        return []

    terms: list[str] = []
    add_unique(terms, [mutation_clean])
    lower = mutation_clean.lower()

    parenthetical_terms = re.findall(r"\(([^)]+)\)", mutation_clean)
    for parenthetical in parenthetical_terms:
        add_unique(terms, [parenthetical])

    mutation_no_parentheses = re.sub(r"\s*\([^)]*\)", "", mutation_clean).strip()
    if mutation_no_parentheses != mutation_clean:
        add_unique(terms, [mutation_no_parentheses])

    # Compound alternatives: S31N/G -> S31N, S31G.
    slash_alt_match = re.fullmatch(r"([A-Z])(\d+)([A-Z](?:/[A-Z])+)" , mutation_no_parentheses.upper())
    if slash_alt_match:
        ref, pos, alts_text = slash_alt_match.groups()
        for alt in alts_text.split("/"):
            if alt:
                add_unique(terms, expand_mutation_terms(f"{ref}{pos}{alt}"))

    # Compound reference/alternative form: I/V27A/T/S -> I27A, I27T, I27S, V27A, V27T, V27S.
    slash_ref_alt_match = re.fullmatch(r"([A-Z](?:/[A-Z])+)\s*(\d+)\s*([A-Z](?:/[A-Z])+)" , mutation_no_parentheses.upper())
    if slash_ref_alt_match:
        refs_text, pos, alts_text = slash_ref_alt_match.groups()
        for ref in refs_text.split("/"):
            for alt in alts_text.split("/"):
                if ref and alt:
                    add_unique(terms, expand_mutation_terms(f"{ref}{pos}{alt}"))

    # Paired HA markers: S123P-496K, S128P-R496K.
    compound_pair_match = re.fullmatch(r"([A-Z]\d+[A-Z])-([A-Z]?\d+[A-Z])", mutation_no_parentheses.upper())
    if compound_pair_match:
        first, second = compound_pair_match.groups()
        add_unique(terms, expand_mutation_terms(first))
        add_unique(terms, [second])
        second_match = re.fullmatch(r"([A-Z]?)(\d+)([A-Z])", second)
        if second_match:
            _ref2, pos2, alt2 = second_match.groups()
            add_unique(terms, [
                f"{alt2}{pos2}",
                f"{pos2}{alt2}",
                f"position {pos2}",
                f"residue {pos2}",
                f"amino acid {pos2}",
            ])

    # Standard substitution: D9N, E627K, D701N.
    substitution_match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation_no_parentheses.upper())
    if substitution_match:
        ref, pos, alt = substitution_match.groups()
        ref3 = AA3.get(ref, ref)
        alt3 = AA3.get(alt, alt)
        add_unique(terms, [
            f"{ref}{pos}{alt}",
            f"{ref3}{pos}{alt3}",
            f"{ref}{pos} {alt}",
            f"{ref3} {pos} {alt3}",
            f"{ref}{pos} to {alt}",
            f"{ref3}{pos} to {alt3}",
            f"{ref}{pos}{alt} substitution",
            f"{ref3}{pos}{alt3} substitution",
            f"{alt}{pos}",
            f"{pos}{alt}",
            f"{alt3}{pos}",
            f"{pos} {alt}",
            f"{pos} {alt3}",
            f"position {pos}",
            f"residue {pos}",
            f"amino acid {pos}",
        ])

    # Residue-only markers: 34R, 199G, P136, Y137.
    residue_marker_match = re.fullmatch(r"(?:([A-Z])\s*)?(\d+)([A-Z])", mutation_no_parentheses.upper())
    if residue_marker_match and not substitution_match:
        ref_or_empty, pos, residue = residue_marker_match.groups()
        residue3 = AA3.get(residue, residue)
        add_unique(terms, [
            f"{pos}{residue}",
            f"{residue}{pos}",
            f"{pos} {residue}",
            f"{pos} {residue3}",
            f"{residue3}{pos}",
            f"position {pos}",
            f"residue {pos}",
            f"amino acid {pos}",
        ])
        if ref_or_empty:
            add_unique(terms, [f"{ref_or_empty}{pos}{residue}"])

    # Deletion range: 80-84del, 59-70del.
    deletion_range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)\s*del", lower)
    if deletion_range_match:
        start, end = deletion_range_match.groups()
        add_unique(terms, [
            f"{start}-{end}del",
            f"{start}-{end} deletion",
            f"deletion {start}-{end}",
            f"deletion at positions {start}-{end}",
            f"positions {start}-{end}",
            f"amino acids {start}-{end}",
        ])

    # Simple deletion: E126del or 126del.
    deletion_simple_match = re.fullmatch(r"([a-z]?)(\d+)del", lower)
    if deletion_simple_match:
        aa, pos = deletion_simple_match.groups()
        add_unique(terms, [
            f"{pos}del",
            f"{pos} deletion",
            f"deletion {pos}",
            f"deletion at position {pos}",
            f"position {pos}",
            f"residue {pos}",
            f"amino acid {pos}",
        ])
        if aa:
            add_unique(terms, [f"{aa.upper()}{pos} deletion"])

    # NA stalk deletion descriptions.
    if "stalk" in lower and "deletion" in lower:
        add_unique(terms, [
            "NA stalk deletion",
            "neuraminidase stalk deletion",
            "stalk deletion",
            "deletion in the NA stalk",
            "deletion in the neuraminidase stalk",
        ])
        aa_deletion_match = re.search(r"(\d+)\s*[- ]?aa", lower)
        if aa_deletion_match:
            aa_count = aa_deletion_match.group(1)
            add_unique(terms, [
                f"{aa_count}-AA deletion",
                f"{aa_count} amino acid deletion",
                f"{aa_count} amino acids deletion",
            ])
        range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", lower)
        if range_match:
            start, end = range_match.groups()
            add_unique(terms, [
                f"{start}-{end} deletion",
                f"deletion {start}-{end}",
                f"positions {start}-{end}",
            ])

    # NS1 PDZ/ESEV motif forms.
    if "esev" in lower or "pdz" in lower:
        add_unique(terms, [
            "ESEV",
            "PDZ",
            "PDZ domain",
            "PDZ-binding motif",
            "PDZ binding motif",
            "C-terminal ESEV",
            "NS1 ESEV",
        ])

    return list(dict.fromkeys(terms))


def build_query(segment: str, mutation: str, fulltext_only: bool = False) -> str:
    """Construct a broad PubMed query for a given segment and mutation.

    PubMed is used only to create candidate PMIDs. Final interpretation is
    performed later from full text whenever full text is available. Therefore
    this query intentionally avoids double-ANDing config.QUERY_TEMPLATE with
    another segment/mutation group, which previously made the search too strict.
    """
    influenza_terms = [
        "Influenza in Birds",
        "avian influenza",
        "avian influenza virus",
        "highly pathogenic avian influenza",
        "low pathogenic avian influenza",
        "HPAI",
        "LPAI",
        "H5N1",
        "H7N9",
        "H5N6",
        "H5N8",
        "H9N2",
        "influenza A virus",
        "IAV",
        "influenza virus",
    ]

    segment_terms = expand_segment_terms(segment)
    mutation_terms = expand_mutation_terms(mutation)

    influenza_group = build_or_group(influenza_terms, "Title/Abstract")
    segment_group = build_or_group(segment_terms, "Title/Abstract")
    mutation_group = build_or_group(mutation_terms, "All Fields")

    query_parts = [influenza_group, segment_group, mutation_group]
    query = " AND ".join(f"({part})" for part in query_parts if part)

    if fulltext_only:
        query += ' AND "free full text"[Filter]'

    return query
