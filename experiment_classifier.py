"""Experiment detection utilities for the mutation search pipeline.

This module implements rule‑based, MeSH‑based and optional LLM‑based
detectors for identifying whether a PubMed record describes an in vivo
(animal) experiment or an in vitro (cell culture) experiment.  It also
provides a helper function that applies the detectors to a single record
and aggregates their outputs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from . import config
from .evidence_extractor import extract_evidence
from .pubmed_client import get_openai_client




def assign_evidence_grade(
    has_mutation: bool,
    has_in_vivo: bool,
    has_in_vitro: bool,
    has_effect: bool,
    confidence: int = 0,
) -> str:
    """Assign evidence grade using structured evidence and experiment flags."""
    if has_mutation and has_in_vivo and has_effect and confidence >= 60:
        return "A"
    if has_mutation and has_in_vitro and has_effect and confidence >= 50:
        return "B"
    if has_mutation and has_effect:
        return "C"
    if has_mutation:
        return "D"
    return "E"


def extract_mutation_effect_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Extract structured mutation evidence using evidence_extractor.py."""
    title = str(record.get("title", "") or "")
    abstract = str(record.get("abstract", "") or "")
    full_text = str(record.get("full_text", "") or "")
    mutation = record.get("mutation") or record.get("aa_change") or record.get("query")

    evidence = extract_evidence(
        title=title,
        abstract=abstract,
        full_text=full_text,
        mutation=str(mutation) if mutation is not None else None,
    )

    vivo_rule = detect_animal_experiment_rule(record)
    vitro_rule = detect_in_vitro_rule(record)

    has_mutation = bool(evidence.get("mutation_sentence"))
    has_effect = bool(evidence.get("effect")) or bool(evidence.get("effect_sentence"))
    has_in_vivo = bool(vivo_rule.get("flag")) or "in vivo" in str(evidence.get("experiment_type", ""))
    has_in_vitro = bool(vitro_rule.get("flag")) or "in vitro" in str(evidence.get("experiment_type", ""))
    confidence = int(evidence.get("confidence", 0) or 0)

    evidence["methods_in_vivo"] = int(has_in_vivo and bool(evidence.get("methods_sentence")))
    evidence["methods_in_vitro"] = int(has_in_vitro and bool(evidence.get("methods_sentence")))
    evidence["methods_in_vivo_terms_found"] = vivo_rule.get("evidence", "") if evidence["methods_in_vivo"] else ""
    evidence["methods_in_vitro_terms_found"] = vitro_rule.get("evidence", "") if evidence["methods_in_vitro"] else ""
    evidence["evidence_grade"] = assign_evidence_grade(
        has_mutation=has_mutation,
        has_in_vivo=has_in_vivo,
        has_in_vitro=has_in_vitro,
        has_effect=has_effect,
        confidence=confidence,
    )

    return evidence


def detect_animal_experiment_rule(record: dict[str, Any]) -> dict[str, Any]:
    """Detect in vivo animal experiments using title/abstract keyword rules.

    The function searches for the co‑occurrence of keywords referring to
    animal species and experiment‑related terms within the combined title
    and abstract.  If both a species and an experiment term are found,
    the record is flagged as an in vivo experiment.

    Returns a dictionary with keys:
      - ``flag``: 1 if an in vivo experiment is detected, else 0
      - ``species``: list of matched animal species
      - ``evidence``: concatenated evidence string describing the hits
    """
    title = record.get("title", "")
    abstract = record.get("abstract", "")
    full_text = record.get("full_text", "")
    text = f"{title} {abstract} {full_text}".lower()

    animal_species_keywords = [
        "mouse", "mice", "murine",
        "rat", "rats",
        "ferret", "ferrets",
        "chicken", "chickens",
        "duck", "ducks",
        "quail",
        "turkey", "turkeys",
        "goose", "geese",
        "pig", "pigs", "swine",
        "guinea pig", "guinea pigs",
        "hamster", "hamsters",
        "macaque", "monkey", "nonhuman primate",
    ]

    experiment_keywords = [
        "infected", "infection",
        "inoculated", "inoculation",
        "challenged", "challenge",
        "viral replication", "replication",
        "pathogenicity", "pathogenesis",
        "virulence",
        "transmission",
        "lethality",
        "mortality",
        "survival",
        "animal model",
        "in vivo",
        "body weight loss",
        "ld50",
        "mld50",
        "eid50",
        "intranasally inoculated",
        "infected intranasally",
        "pathogenicity in mice",
        "virulence in mice",
    ]

    species_hits = [k for k in animal_species_keywords if k in text]
    experiment_hits = [k for k in experiment_keywords if k in text]
    is_in_vivo = int(bool(species_hits) and bool(experiment_hits))
    evidence_parts: list[str] = []
    if species_hits:
        evidence_parts.append("species=" + ",".join(species_hits))
    if experiment_hits:
        evidence_parts.append("experiment=" + ",".join(experiment_hits))
    return {
        "flag": is_in_vivo,
        "species": species_hits,
        "evidence": " | ".join(evidence_parts),
    }


def detect_animal_experiment_mesh(record: dict[str, Any]) -> dict[str, Any]:
    """Detect in vivo animal experiments using MeSH terms.

    The function inspects the list of MeSH terms associated with a record
    for entries related to animals and infection/replication.  Both a
    species‑related term and an experiment‑related term must be present
    to flag the record as an in vivo experiment.
    """
    mesh_terms: list[str] = record.get("mesh_terms", [])  # type: ignore
    mesh_text = " ".join(mesh_terms).lower()
    mesh_animal_terms = [
        "animals",
        "mice",
        "rats",
        "ferrets",
        "chickens",
        "ducks",
        "swine",
        "disease models, animal",
        "animal experimentation",
    ]
    mesh_experiment_terms = [
        "disease models, animal",
        "animal experimentation",
        "infection",
        "virus replication",
        "virulence",
        "pathogenicity",
        "disease transmission, infectious",
    ]
    animal_hits = [k for k in mesh_animal_terms if k in mesh_text]
    experiment_hits = [k for k in mesh_experiment_terms if k in mesh_text]
    is_in_vivo = int(bool(animal_hits) and bool(experiment_hits))
    evidence_parts: list[str] = []
    if animal_hits:
        evidence_parts.append("mesh_animal=" + ",".join(animal_hits))
    if experiment_hits:
        evidence_parts.append("mesh_experiment=" + ",".join(experiment_hits))
    return {
        "flag": is_in_vivo,
        "evidence": " | ".join(evidence_parts),
    }


def detect_animal_experiment_llm(record: dict[str, Any], client: Any) -> dict[str, Any]:
    """Use an LLM to classify whether a record describes an in vivo experiment.

    The LLM is only used when both rule‑ and MeSH‑based detectors have
    failed to identify an in vivo experiment (i.e., both returned flag=0).
    The prompt instructs the model to look for evidence of experimental
    infection, inoculation, challenge, replication or pathogenicity tests
    in animals.

    Returns a dictionary with keys ``flag`` and ``evidence``.  When the
    client is unavailable, the function returns empty strings for these
    fields along with a message indicating that the LLM was skipped.
    """
    if client is None:
        return {
            "flag": "",
            "evidence": "LLM skipped: OPENAI_API_KEY not set or openai package not installed",
        }
    pmid = record.get("PMID", "")
    title = record.get("title", "")
    abstract = record.get("abstract", "")
    full_text = record.get("full_text", "")
    full_text_excerpt = full_text[:6000]

    if not title and not abstract and not full_text:
        return {"flag": 0, "evidence": "No title/abstract/full_text available"}
    prompt = f"""
Classify whether this PubMed record describes an in vivo animal experiment.

IMPORTANT:
Do NOT classify review articles as animal experiments.

Use only the title, abstract, and available full-text excerpt.

Return JSON only with keys:
- animal_experiment: 1 or 0
- species: list of animal species if mentioned
- evidence: short reason, maximum 30 words

Definition:
animal_experiment = 1 only if the title/abstract indicate experimental use of animals,
such as infection, inoculation, challenge, transmission, pathogenicity, virulence,
replication, survival, or mortality testing in animals.

Do not count:
- pure sequence analysis
- surveillance only
- human clinical cases only
- review articles without original animal experiments
- cell culture only

PMID: {pmid}
Title: {title}
Abstract: {abstract}
Full text excerpt: {full_text_excerpt}
"""
    try:
        response = client.responses.create(model=config.LLM_MODEL, input=prompt)
        text = response.output_text.strip()
        try:
            parsed = json.loads(text)
            flag = int(parsed.get("animal_experiment", 0))
            species = parsed.get("species", [])
            evidence = parsed.get("evidence", "")
            return {
                "flag": flag,
                "evidence": f"species={species}; evidence={evidence}",
            }
        except Exception:
            return {
                "flag": "",
                "evidence": f"LLM returned non-JSON: {text[:200]}",
            }
    except Exception as exc:
        return {
            "flag": "",
            "evidence": f"LLM error: {exc}",
        }


def detect_in_vitro_rule(record: dict[str, Any]) -> dict[str, Any]:
    """Detect in vitro (cell culture) experiments using keyword rules.

    The function searches for the co‑occurrence of cell culture keywords and
    experiment‑related terms within the combined title and abstract.  If both
    a cell term and an experiment term are found, the record is flagged as
    an in vitro experiment.
    """
    title = record.get("title", "")
    abstract = record.get("abstract", "")
    full_text = record.get("full_text", "")

    text = f"{title} {abstract} {full_text}".lower()
    cell_keywords = [
        "in vitro", "cell culture", "cell line", "cell lines", "cells",
        "mdck", "vero", "293t", "hepg2", "hek-293", "hek293", "a549",
        "primary cells", "mndck", "cpe", "caco-2", "calu-3", "huh7",
        "df-1", "cef", "chicken embryo fibroblast", "minigenome",
    ]
    experiment_keywords = [
        "infected", "infection",
        "inoculated", "inoculation",
        "challenged", "challenge",
        "replication", "viral replication", "virus replication",
        "viral yield", "plaque assay", "cytopathic", "viral growth",
        "propagation", "titers", "virus titers",
        "tcid50", "western blot", "immunofluorescence", "luciferase assay",
        "minigenome assay", "replication kinetics", "growth curve",
    ]
    cell_hits = [k for k in cell_keywords if k in text]
    exp_hits = [k for k in experiment_keywords if k in text]
    is_in_vitro = int(bool(cell_hits) and bool(exp_hits))
    evidence_parts: list[str] = []
    if cell_hits:
        evidence_parts.append("cells=" + ",".join(cell_hits))
    if exp_hits:
        evidence_parts.append("experiment=" + ",".join(exp_hits))
    return {
        "flag": is_in_vitro,
        "cell_types": cell_hits,
        "evidence": " | ".join(evidence_parts),
    }


def detect_in_vitro_mesh(record: dict[str, Any]) -> dict[str, Any]:
    """Detect in vitro experiments using MeSH terms.

    The function examines MeSH terms for entries related to cell culture and
    viral infection or replication.  Both a cell‑culture term and an
    experiment‑related term must be present to flag the record.
    """
    mesh_terms: list[str] = record.get("mesh_terms", [])  # type: ignore
    mesh_text = " ".join(mesh_terms).lower()
    mesh_cell_terms = [
        "cells, cultured", "cell line", "cell lines", "cell culture",
        "vero cells", "mdck cells", "hela cells", "hep g2 cells",
    ]
    mesh_experiment_terms = [
        "virus replication", "infection", "virus cultivation",
        "viral plaque assay", "virus growth", "virus titers",
    ]
    cell_hits = [k for k in mesh_cell_terms if k in mesh_text]
    exp_hits = [k for k in mesh_experiment_terms if k in mesh_text]
    is_in_vitro = int(bool(cell_hits) and bool(exp_hits))
    evidence_parts: list[str] = []
    if cell_hits:
        evidence_parts.append("mesh_cells=" + ",".join(cell_hits))
    if exp_hits:
        evidence_parts.append("mesh_experiment=" + ",".join(exp_hits))
    return {
        "flag": is_in_vitro,
        "evidence": " | ".join(evidence_parts),
    }


def detect_in_vitro_llm(record: dict[str, Any], client: Any) -> dict[str, Any]:
    """Use an LLM to classify whether a record describes an in vitro experiment.

    The LLM is invoked only when rule‑ and MeSH‑based detectors have not
    identified an in vitro experiment.  The prompt instructs the model to
    look for evidence of cell culture infection or replication.
    """
    if client is None:
        return {
            "flag": "",
            "evidence": "LLM skipped: OPENAI_API_KEY not set or openai package not installed",
        }
    pmid = record.get("PMID", "")
    title = record.get("title", "")
    abstract = record.get("abstract", "")
    full_text = record.get("full_text", "")
    full_text_excerpt = full_text[:6000]

    if not title and not abstract and not full_text:
        return {"flag": 0, "cell_types": [], "evidence": "No title/abstract/full_text available"}
    prompt = f"""
Classify whether this PubMed record describes an in vitro cell‑culture experiment.

IMPORTANT:
Do NOT classify review articles as experiments.

Use only the title, abstract, and available full-text excerpt.

Return JSON only with keys:
- in_vitro_experiment: 1 or 0
- cell_types: list of cell types if mentioned
- evidence: short reason, maximum 30 words

Definition:
in_vitro_experiment = 1 only if the title/abstract indicate experimental use of
cell culture or cell lines, such as infection, inoculation, replication,
propagation, plaque assays or growth of influenza virus in cells.

Do not count:
- pure sequence analysis
- surveillance only
- human clinical cases only
- review articles without original cell culture experiments
- in vivo animal experiments

PMID: {pmid}
Title: {title}
Abstract: {abstract}
Full text excerpt: {full_text_excerpt}
"""
    try:
        response = client.responses.create(model=config.LLM_MODEL, input=prompt)
        text = response.output_text.strip()
        try:
            parsed = json.loads(text)
            flag = int(parsed.get("in_vitro_experiment", 0))
            cell_types = parsed.get("cell_types", [])
            evidence = parsed.get("evidence", "")
            return {
                "flag": flag,
                "cell_types": cell_types,
                "evidence": f"cell_types={cell_types}; evidence={evidence}",
            }
        except Exception:
            return {
                "flag": "",
                "evidence": f"LLM returned non-JSON: {text[:200]}",
            }
    except Exception as exc:
        return {
            "flag": "",
            "evidence": f"LLM error: {exc}",
        }


def analyze_one_pubmed_detail(detail: dict[str, Any], llm_client: Any = None) -> dict[str, Any]:
    """Analyse one PubMed record for in vivo and in vitro experiments.

    This helper function applies the rule‑ and MeSH‑based detectors for both
    in vivo and in vitro categories.  If neither detector yields a positive
    result and an OpenAI client is provided, an LLM is invoked for the
    category.  The LLM is skipped when rule or MeSH detectors have already
    flagged the record.

    Returns a dictionary with keys:
      - ``pmid``: the PubMed identifier
      - ``vivo_rule_result`` / ``vivo_mesh_result`` / ``vivo_llm_result``
      - ``vitro_rule_result`` / ``vitro_mesh_result`` / ``vitro_llm_result``
    """
    pmid = detail.get("PMID", "")
    mutation_effect_summary = extract_mutation_effect_summary(detail)

    # In vivo detectors
    vivo_rule = detect_animal_experiment_rule(detail)
    vivo_mesh = detect_animal_experiment_mesh(detail)
    if llm_client is not None and vivo_rule["flag"] == 0 and vivo_mesh["flag"] == 0:
        vivo_llm = detect_animal_experiment_llm(detail, llm_client)
    else:
        vivo_llm = {
            "flag": "",
            "evidence": "LLM skipped: rule/mesh already positive or LLM disabled",
        }
    # In vitro detectors
    vitro_rule = detect_in_vitro_rule(detail)
    vitro_mesh = detect_in_vitro_mesh(detail)
    if llm_client is not None and vitro_rule["flag"] == 0 and vitro_mesh["flag"] == 0:
        vitro_llm = detect_in_vitro_llm(detail, llm_client)
    else:
        vitro_llm = {
            "flag": "",
            "evidence": "LLM skipped: rule/mesh already positive or LLM disabled",
        }
    return {
        "pmid": pmid,
        "vivo_rule_result": vivo_rule,
        "vivo_mesh_result": vivo_mesh,
        "vivo_llm_result": vivo_llm,
        "vitro_rule_result": vitro_rule,
        "vitro_mesh_result": vitro_mesh,
        "vitro_llm_result": vitro_llm,
        "mutation_effect_summary": mutation_effect_summary,
    }
