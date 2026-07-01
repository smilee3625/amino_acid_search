"""Evidence extraction utilities for the PubMed mutation search pipeline.

This module converts paper text into structured evidence fields. It is
intended to be used by ``experiment_classifier.py`` so that sentence
extraction, host/cell detection, effect detection, and confidence scoring
are kept separate from rule-based in vivo/in vitro classification.
"""

from __future__ import annotations

import re
from typing import Any


SECTION_HEADERS = {
    "methods": [
        "materials and methods",
        "methods",
        "experimental procedures",
        "methodology",
        "virus and cells",
        "animal experiments",
        "ethics statement",
    ],
    "results": [
        "results",
    ],
    "discussion": [
        "discussion",
        "conclusion",
        "conclusions",
    ],
}

SECTION_END_HEADERS = [
    "materials and methods",
    "methods",
    "experimental procedures",
    "methodology",
    "virus and cells",
    "animal experiments",
    "ethics statement",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgments",
    "acknowledgements",
]

HOST_TERMS = {
    "mouse": ["mouse", "mice", "murine"],
    "ferret": ["ferret", "ferrets"],
    "chicken": ["chicken", "chickens"],
    "duck": ["duck", "ducks"],
    "quail": ["quail", "quails"],
    "guinea pig": ["guinea pig", "guinea pigs"],
    "human": ["human", "humans", "patient", "patients"],
    "avian": ["avian", "bird", "birds", "poultry"],
}

CELL_LINE_TERMS = [
    "MDCK",
    "A549",
    "293T",
    "HEK293T",
    "DF-1",
    "CEF",
    "chicken embryo fibroblast",
    "Vero",
    "Calu-3",
    "Caco-2",
    "Huh7",
    "primary cells",
]

IN_VIVO_TERMS = [
    "in vivo",
    "mouse model",
    "mice",
    "murine",
    "ferret",
    "guinea pig",
    "chicken",
    "duck",
    "quail",
    "animal experiment",
    "animal study",
    "challenge experiment",
    "challenge study",
    "intranasally inoculated",
    "infected intranasally",
    "pathogenicity in mice",
    "virulence in mice",
    "lethality",
    "survival curve",
    "body weight loss",
    "LD50",
    "MLD50",
    "EID50",
]

IN_VITRO_TERMS = [
    "in vitro",
    "cell culture",
    "cultured cells",
    "MDCK",
    "A549",
    "293T",
    "HEK293T",
    "DF-1",
    "CEF",
    "Vero",
    "Calu-3",
    "plaque assay",
    "viral replication",
    "replication kinetics",
    "growth curve",
    "TCID50",
    "western blot",
    "immunofluorescence",
    "luciferase assay",
    "minigenome assay",
]

EFFECT_PATTERNS = {
    "virulence_increase": [
        "increased virulence",
        "enhanced virulence",
        "more virulent",
        "increased pathogenicity",
        "enhanced pathogenicity",
    ],
    "virulence_decrease": [
        "attenuated",
        "reduced virulence",
        "decreased virulence",
        "reduced pathogenicity",
        "decreased pathogenicity",
    ],
    "replication_increase": [
        "increased replication",
        "enhanced replication",
        "higher replication",
        "increased viral growth",
        "enhanced viral growth",
        "higher viral titers",
    ],
    "replication_decrease": [
        "reduced replication",
        "decreased replication",
        "lower replication",
        "reduced viral growth",
        "lower viral titers",
    ],
    "host_adaptation": [
        "host adaptation",
        "mammalian adaptation",
        "adaptation to mammals",
    ],
    "polymerase_activity": [
        "polymerase activity",
        "increased polymerase activity",
        "enhanced polymerase activity",
        "reduced polymerase activity",
    ],
    "transmission": [
        "transmission",
        "airborne transmission",
        "contact transmission",
    ],
    "drug_resistance": [
        "drug resistance",
        "oseltamivir resistance",
        "antiviral resistance",
    ],
    "immune_escape": [
        "immune escape",
        "antigenic change",
        "antigenic drift",
    ],
    "receptor_binding": [
        "receptor binding",
        "alpha-2,6",
        "α2,6",
        "alpha-2,3",
        "α2,3",
    ],
    "thermal_stability": [
        "thermal stability",
        "stability at low ph",
        "acid stability",
    ],
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

MAX_SENTENCE_CHARS = 600
MAX_JOINED_FIELD_CHARS = 3000

def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units and prevent paragraph-length hits."""
    text = normalize_text(text)
    if not text:
        return []

    # Split by common sentence boundaries first. Some parsed XML/PDF text
    # lacks normal punctuation, so oversized blocks are split again by
    # semicolon, colon, and bullet-like boundaries.
    rough_units = re.split(r"(?<=[.!?])\s+|\s+[•·]\s+", text)
    sentence_units: list[str] = []

    for unit in rough_units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) <= MAX_SENTENCE_CHARS:
            sentence_units.append(unit)
            continue

        smaller_units = re.split(r"(?<=[;:])\s+|\s+-\s+", unit)
        for smaller in smaller_units:
            smaller = smaller.strip()
            if not smaller:
                continue
            sentence_units.append(truncate_text(smaller, MAX_SENTENCE_CHARS))

    return sentence_units


def truncate_text(text: str, max_chars: int = MAX_SENTENCE_CHARS) -> str:
    """Truncate long extracted text while keeping it readable."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ...[truncated]"


def join_limited(items: list[str], sep: str = " | ", max_chars: int = MAX_JOINED_FIELD_CHARS) -> str:
    """Join extracted sentences without creating very long cells."""
    cleaned = [truncate_text(item) for item in items if normalize_text(item)]
    joined = sep.join(cleaned)
    return truncate_text(joined, max_chars)


def normalize_text(text: Any) -> str:
    """Normalize missing values and whitespace."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def find_terms(text: str, terms: list[str]) -> list[str]:
    """Return case-insensitive term hits."""
    text_lower = normalize_text(text).lower()
    return sorted({term for term in terms if term.lower() in text_lower})


def flatten_effect_terms() -> list[str]:
    """Return all effect terms as a single list."""
    terms: list[str] = []
    for values in EFFECT_PATTERNS.values():
        terms.extend(values)
    return terms


def expand_mutation_terms(mutation: str | None) -> list[str]:
    """Expand a mutation string to common one-letter and three-letter variants."""
    if not mutation:
        return []

    mutation_clean = str(mutation).strip()
    if not mutation_clean:
        return []

    terms = [mutation_clean]
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
        ])

    deletion_range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)\s*del", mutation_clean.lower())
    if deletion_range_match:
        start, end = deletion_range_match.groups()
        terms.extend([
            f"{start}-{end}del",
            f"{start}-{end} deletion",
            f"deletion {start}-{end}",
            f"deletion at positions {start}-{end}",
        ])

    return list(dict.fromkeys(terms))


def extract_section(full_text: str, section_name: str) -> str:
    """Extract a rough section by header name."""
    text = normalize_text(full_text)
    lower = text.lower()
    headers = SECTION_HEADERS.get(section_name, [])

    start_positions = [lower.find(header) for header in headers if lower.find(header) != -1]
    if not start_positions:
        return ""

    start = min(start_positions)
    end_positions = [
        lower.find(header, start + 100)
        for header in SECTION_END_HEADERS
        if lower.find(header, start + 100) != -1
    ]

    if end_positions:
        return text[start:min(end_positions)]
    return text[start:]


def extract_sentences_with_terms(text: str, terms: list[str], limit: int = 3) -> list[str]:
    """Return sentences containing any target term."""
    hits: list[str] = []
    for sentence in split_sentences(text):
        if find_terms(sentence, terms):
            hits.append(truncate_text(sentence))
        if len(hits) >= limit:
            break
    return hits


def extract_mutation_sentences(text: str, mutation: str | None, limit: int = 3) -> list[str]:
    """Return sentences containing mutation spelling variants."""
    mutation_terms = expand_mutation_terms(mutation)
    if not mutation_terms:
        return []
    return extract_sentences_with_terms(text, mutation_terms, limit=limit)


def detect_hosts(text: str) -> list[str]:
    """Detect host species or host context terms."""
    text_lower = normalize_text(text).lower()
    hits: list[str] = []
    for host, terms in HOST_TERMS.items():
        if any(term.lower() in text_lower for term in terms):
            hits.append(host)
    return hits


def detect_cell_lines(text: str) -> list[str]:
    """Detect cell line terms."""
    return find_terms(text, CELL_LINE_TERMS)


def detect_effects(text: str) -> list[str]:
    """Detect normalized biological effect categories."""
    text_lower = normalize_text(text).lower()
    effects: list[str] = []
    for effect, terms in EFFECT_PATTERNS.items():
        if any(term.lower() in text_lower for term in terms):
            effects.append(effect)
    return effects


def detect_experiment_types(text: str) -> list[str]:
    """Detect broad experiment types from evidence text."""
    types: list[str] = []
    if find_terms(text, IN_VIVO_TERMS):
        types.append("in vivo")
    if find_terms(text, IN_VITRO_TERMS):
        types.append("in vitro")
    return types


def score_confidence(
    mutation_sentences: list[str],
    effect_sentences: list[str],
    methods_sentences: list[str],
    results_sentences: list[str],
    hosts: list[str],
    cell_lines: list[str],
    experiment_types: list[str],
) -> int:
    """Score evidence confidence on a 0-100 scale using transparent rules."""
    score = 0

    if mutation_sentences:
        score += 25
    if effect_sentences:
        score += 20
    if methods_sentences:
        score += 15
    if results_sentences:
        score += 15
    if hosts:
        score += 10
    if cell_lines:
        score += 10
    if experiment_types:
        score += 5

    return min(score, 100)


def extract_evidence(
    *,
    title: str = "",
    abstract: str = "",
    full_text: str = "",
    mutation: str | None = None,
) -> dict[str, Any]:
    """Extract structured mutation evidence from paper text.

    Parameters
    ----------
    title:
        Paper title.

    abstract:
        Paper abstract.

    full_text:
        Full text when available. Empty string is allowed.

    mutation:
        Mutation or motif query, such as ``D701N`` or ``80-84del``.

    Returns
    -------
    dict:
        Structured evidence fields ready to be merged into downstream output.
    """
    title = normalize_text(title)
    abstract = normalize_text(abstract)
    full_text = normalize_text(full_text)

    combined_text = f"{title} {abstract} {full_text}".strip()
    methods_text = extract_section(full_text, "methods")
    results_text = extract_section(full_text, "results")
    discussion_text = extract_section(full_text, "discussion")

    mutation_sentences = extract_mutation_sentences(combined_text, mutation)
    effect_terms = flatten_effect_terms()
    effect_sentences = extract_sentences_with_terms(combined_text, effect_terms, limit=3)

    methods_sentences = extract_sentences_with_terms(
        methods_text,
        IN_VIVO_TERMS + IN_VITRO_TERMS + expand_mutation_terms(mutation),
        limit=3,
    )
    results_sentences = extract_sentences_with_terms(
        results_text,
        effect_terms + expand_mutation_terms(mutation),
        limit=3,
    )
    discussion_sentences = extract_sentences_with_terms(
        discussion_text,
        effect_terms + expand_mutation_terms(mutation),
        limit=3,
    )

    evidence_scope_text = " ".join(
        mutation_sentences
        + effect_sentences
        + methods_sentences
        + results_sentences
        + discussion_sentences
    )
    if not evidence_scope_text:
        evidence_scope_text = combined_text

    hosts = detect_hosts(evidence_scope_text)
    cell_lines = detect_cell_lines(evidence_scope_text)
    effects = detect_effects(evidence_scope_text)
    experiment_types = detect_experiment_types(evidence_scope_text)
    confidence = score_confidence(
        mutation_sentences=mutation_sentences,
        effect_sentences=effect_sentences,
        methods_sentences=methods_sentences,
        results_sentences=results_sentences,
        hosts=hosts,
        cell_lines=cell_lines,
        experiment_types=experiment_types,
    )

    return {
        "mutation_sentence": join_limited(mutation_sentences),
        "effect_sentence": join_limited(effect_sentences),
        "methods_sentence": join_limited(methods_sentences),
        "results_sentence": join_limited(results_sentences),
        "discussion_sentence": join_limited(discussion_sentences),
        "host": ";".join(hosts),
        "cell_line": ";".join(cell_lines),
        "experiment_type": ";".join(experiment_types),
        "effect": ";".join(effects),
        "confidence": confidence,
        "effect_terms_found": ";".join(find_terms(combined_text, effect_terms)),
    }