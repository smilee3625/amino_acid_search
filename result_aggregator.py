"""Aggregate search and classification results for each mutation.

This module provides functions to execute a PubMed search for a given
segment/mutation pair, fetch detailed records, apply experiment
classification detectors and summarise the results into a dictionary ready
for tabulation.  The logic closely follows the original ``search_pubmed``
function but is refactored into a standalone module for improved
maintainability and testability.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm  # type: ignore

from . import config
from .query_builder import build_query
from .pubmed_client import (
    run_pubmed_esearch,
    fetch_pubmed_details,
    get_best_fulltext,
    is_review_article,
)
from .experiment_classifier import analyze_one_pubmed_detail

def search_and_analyze_mutation(
    segment: str,
    mutation: str,
    llm_client: Any = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Search PubMed for a mutation and aggregate in vivo/in vitro results.

    Parameters
    ----------
    segment:
        Influenza gene segment (e.g., "PB2", "HA").

    mutation:
        Amino acid change or motif (e.g., "E627K").

    llm_client:
        Optional OpenAI client for LLM‑based classification.  Pass ``None``
        to disable LLM classification.

    show_progress:
        If true, a progress bar is displayed when analysing PMIDs.  This
        parameter is mainly intended for command‑line use.  When running
        automated tests it can be set to ``False`` to suppress output.

    Returns
    -------
    dict:
        A dictionary keyed by the names defined in ``config.RESULT_COLUMNS``
        containing counts, flags, lists of PMIDs and evidence strings.
    """
    # Construct the query
    query = build_query(segment, mutation)
    result: Dict[str, Any] = {
        "segment": segment,
        "mutation": mutation,
        "query": query,
        "query_count": 0,
        "analyzed_paper_count": 0,
        "in_vivo_paper_count": 0,
        "in_vitro_paper_count": 0,
        "review_paper_count": 0,
        "PMID_list": "",
        "review_pmid_list": "",
        "full_text_available": 0,
        "full_text_paper_count": 0,
        "full_text_pmids": "",
        "full_text_source": "",
        "full_text_status": "NOT_FOUND",
        "no_full_text_paper_count": 0,
        "no_full_text_pmids": "",
        "in_vivo_pmids": "",
        "in_vitro_pmids": "",
        "in_vivo_experiment_rule": 0,
        "in_vivo_experiment_mesh": 0,
        "in_vivo_experiment_llm": "",
        "in_vivo_experiment_final": 0,
        "in_vivo_species_detected": "",
        "in_vivo_evidence_rule": "",
        "in_vivo_evidence_mesh": "",
        "in_vivo_evidence_llm": "",
        "in_vitro_experiment_rule": 0,
        "in_vitro_experiment_mesh": 0,
        "in_vitro_experiment_llm": "",
        "in_vitro_experiment_final": 0,
        "in_vitro_cell_types_detected": "",
        "in_vitro_evidence_rule": "",
        "in_vitro_evidence_mesh": "",
        "in_vitro_evidence_llm": "",
        "mutation_sentence": "",
        "effect_sentence": "",
        "methods_sentence": "",
        "results_sentence": "",
        "discussion_sentence": "",
        "host": "",
        "cell_line": "",
        "experiment_type": "",
        "effect": "",
        "confidence": 0,
        "effect_terms_found": "",
        "methods_in_vivo": 0,
        "methods_in_vitro": 0,
        "methods_in_vivo_terms_found": "",
        "methods_in_vitro_terms_found": "",
        "evidence_grade": "E",
        "error": "",
    }
    try:
        # Execute the search
        count, id_list = run_pubmed_esearch(query)
        result["query_count"] = count
        result["PMID_list"] = ";".join(id_list)

        # Fetch details for a limited number of PMIDs.
        # Review articles are counted and listed, then excluded from
        # downstream original-experiment analysis.
        detail_pmids = id_list[: config.MAX_PMIDS_FOR_DETAIL_CHECK]
        all_details = fetch_pubmed_details(detail_pmids)

        review_pmids: List[str] = []
        filtered_details: List[Dict[str, Any]] = []

        for detail in all_details:
            if is_review_article(detail):
                pmid = str(detail.get("PMID", "")).strip()
                if pmid:
                    review_pmids.append(pmid)
            else:
                filtered_details.append(detail)

        details = filtered_details
        review_pmids_unique = sorted(set(review_pmids))

        result["review_paper_count"] = len(review_pmids_unique)
        result["review_pmid_list"] = ";".join(review_pmids_unique)
        result["analyzed_paper_count"] = len(details)


        # Fetch full text for open-access records, when available.
        # Priority is handled inside get_best_fulltext(): PMC -> Europe PMC -> Unpaywall.
        # Full text is stored only in memory for classification and is not written to Excel.
        full_text_pmids: List[str] = []
        full_text_sources: List[str] = []
        full_text_statuses: List[str] = []

        for detail in details:
            pmid = str(detail.get("PMID", ""))
            detail["mutation"] = mutation
            detail["segment"] = segment
            detail["query"] = query

            full_text, source, status = get_best_fulltext(pmid)

            detail["full_text"] = full_text
            detail["full_text_source"] = source

            full_text_statuses.append(status)

            if full_text:
                full_text_pmids.append(pmid)

            if source:
                full_text_sources.append(source)

        full_text_pmids_unique = sorted(set(full_text_pmids))
        full_text_sources_unique = sorted(set(s for s in full_text_sources if s))

        result["full_text_available"] = int(bool(full_text_pmids_unique))
        result["full_text_paper_count"] = len(full_text_pmids_unique)
        result["full_text_pmids"] = ";".join(full_text_pmids_unique)
        result["full_text_source"] = ";".join(full_text_sources_unique)

        if full_text_pmids_unique:
            result["full_text_status"] = "SUCCESS"
        elif "ERROR" in full_text_statuses:
            result["full_text_status"] = "ERROR"
        else:
            result["full_text_status"] = "NOT_FOUND"

        # Force downstream interpretation to use full text only.
        # Papers without retrieved full text are kept out of in-vivo/in-vitro
        # classification so the final calls are not made from title/abstract alone.
        no_full_text_pmids = [
            str(detail.get("PMID", ""))
            for detail in details
            if not detail.get("full_text")
        ]
        details = [detail for detail in details if detail.get("full_text")]

        result["no_full_text_paper_count"] = len([pmid for pmid in no_full_text_pmids if pmid])
        result["no_full_text_pmids"] = ";".join(sorted(set(pmid for pmid in no_full_text_pmids if pmid)))
        result["analyzed_paper_count"] = len(details)

        # Initialize accumulators
        vivo_rule_flags: List[int] = []
        vivo_mesh_flags: List[int] = []
        vivo_llm_flags: List[int] = []
        vivo_species_all: List[str] = []
        vivo_evidence_rule_all: List[str] = []
        vivo_evidence_mesh_all: List[str] = []
        vivo_evidence_llm_all: List[str] = []
        vitro_rule_flags: List[int] = []
        vitro_mesh_flags: List[int] = []
        vitro_llm_flags: List[int] = []
        vitro_cell_all: List[str] = []
        vitro_evidence_rule_all: List[str] = []
        vitro_evidence_mesh_all: List[str] = []
        vitro_evidence_llm_all: List[str] = []
        vivo_pmid_hits: List[str] = []
        vitro_pmid_hits: List[str] = []
        mutation_sentence_all: List[str] = []
        effect_sentence_all: List[str] = []
        # Additional accumulators for new fields
        methods_sentence_all: List[str] = []
        results_sentence_all: List[str] = []
        discussion_sentence_all: List[str] = []
        host_all: List[str] = []
        cell_line_all: List[str] = []
        experiment_type_all: List[str] = []
        effect_all: List[str] = []
        confidence_scores: List[int] = []
        effect_terms_all: List[str] = []
        methods_vivo_flags: List[int] = []
        methods_vitro_flags: List[int] = []
        methods_vivo_terms_all: List[str] = []
        methods_vitro_terms_all: List[str] = []
        evidence_grades_all: List[str] = []
        # Analyse each detail concurrently
        MAX_LLM_WORKERS = 5
        with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
            futures = [
                executor.submit(analyze_one_pubmed_detail, detail, llm_client)
                for detail in details
            ]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=f"{segment} {mutation} PMID analysis", unit="PMID", leave=False)
            for future in iterator:
                analysis = future.result()
                pmid = analysis["pmid"]
                vivo_rule_result = analysis["vivo_rule_result"]
                vivo_mesh_result = analysis["vivo_mesh_result"]
                vivo_llm_result = analysis["vivo_llm_result"]
                vitro_rule_result = analysis["vitro_rule_result"]
                vitro_mesh_result = analysis["vitro_mesh_result"]
                vitro_llm_result = analysis["vitro_llm_result"]
                mutation_effect_summary = analysis.get("mutation_effect_summary", {})

                if mutation_effect_summary.get("mutation_sentence"):
                    mutation_sentence_all.append(f"{pmid}: {mutation_effect_summary['mutation_sentence']}")
                if mutation_effect_summary.get("effect_sentence"):
                    effect_sentence_all.append(f"{pmid}: {mutation_effect_summary['effect_sentence']}")
                # Additional evidence field handling
                if mutation_effect_summary.get("methods_sentence"):
                    methods_sentence_all.append(f"{pmid}: {mutation_effect_summary['methods_sentence']}")
                if mutation_effect_summary.get("results_sentence"):
                    results_sentence_all.append(f"{pmid}: {mutation_effect_summary['results_sentence']}")
                if mutation_effect_summary.get("discussion_sentence"):
                    discussion_sentence_all.append(f"{pmid}: {mutation_effect_summary['discussion_sentence']}")
                if mutation_effect_summary.get("host"):
                    host_all.extend([x.strip() for x in str(mutation_effect_summary["host"]).split(";") if x.strip()])
                if mutation_effect_summary.get("cell_line"):
                    cell_line_all.extend([x.strip() for x in str(mutation_effect_summary["cell_line"]).split(";") if x.strip()])
                if mutation_effect_summary.get("experiment_type"):
                    experiment_type_all.extend([x.strip() for x in str(mutation_effect_summary["experiment_type"]).split(";") if x.strip()])
                if mutation_effect_summary.get("effect"):
                    effect_all.extend([x.strip() for x in str(mutation_effect_summary["effect"]).split(";") if x.strip()])
                if mutation_effect_summary.get("confidence") is not None:
                    confidence_scores.append(int(mutation_effect_summary.get("confidence", 0)))
                if mutation_effect_summary.get("effect_terms_found"):
                    effect_terms_all.extend(
                        [term.strip() for term in mutation_effect_summary["effect_terms_found"].split(";") if term.strip()]
                    )
                if mutation_effect_summary.get("methods_in_vivo"):
                    methods_vivo_flags.append(1)
                if mutation_effect_summary.get("methods_in_vitro"):
                    methods_vitro_flags.append(1)
                if mutation_effect_summary.get("methods_in_vivo_terms_found"):
                    methods_vivo_terms_all.extend(
                        [term.strip() for term in mutation_effect_summary["methods_in_vivo_terms_found"].split(";") if term.strip()]
                    )
                if mutation_effect_summary.get("methods_in_vitro_terms_found"):
                    methods_vitro_terms_all.extend(
                        [term.strip() for term in mutation_effect_summary["methods_in_vitro_terms_found"].split(";") if term.strip()]
                    )
                if mutation_effect_summary.get("evidence_grade"):
                    evidence_grades_all.append(str(mutation_effect_summary["evidence_grade"]))

                # accumulate flags and pmids for in vivo
                if vivo_rule_result["flag"] == 1:
                    vivo_rule_flags.append(1)
                if vivo_mesh_result["flag"] == 1:
                    vivo_mesh_flags.append(1)
                if vivo_llm_result.get("flag") == 1:
                    vivo_llm_flags.append(1)
                if vivo_rule_result.get("species"):
                    vivo_species_all.extend(vivo_rule_result.get("species", []))
                if any([
                    vivo_rule_result["flag"] == 1,
                    vivo_mesh_result["flag"] == 1,
                    vivo_llm_result.get("flag") == 1,
                ]):
                    vivo_pmid_hits.append(pmid)
                if vivo_rule_result.get("evidence"):
                    vivo_evidence_rule_all.append(f"{pmid}: {vivo_rule_result['evidence']}")
                if vivo_mesh_result.get("evidence"):
                    vivo_evidence_mesh_all.append(f"{pmid}: {vivo_mesh_result['evidence']}")
                if vivo_llm_result.get("evidence"):
                    vivo_evidence_llm_all.append(f"{pmid}: {vivo_llm_result['evidence']}")
                # accumulate flags and pmids for in vitro
                if vitro_rule_result["flag"] == 1:
                    vitro_rule_flags.append(1)
                if vitro_mesh_result["flag"] == 1:
                    vitro_mesh_flags.append(1)
                if vitro_llm_result.get("flag") == 1:
                    vitro_llm_flags.append(1)
                if vitro_rule_result.get("cell_types"):
                    vitro_cell_all.extend(vitro_rule_result.get("cell_types", []))
                if any([
                    vitro_rule_result["flag"] == 1,
                    vitro_mesh_result["flag"] == 1,
                    vitro_llm_result.get("flag") == 1,
                ]):
                    vitro_pmid_hits.append(pmid)
                if vitro_rule_result.get("evidence"):
                    vitro_evidence_rule_all.append(f"{pmid}: {vitro_rule_result['evidence']}")
                if vitro_mesh_result.get("evidence"):
                    vitro_evidence_mesh_all.append(f"{pmid}: {vitro_mesh_result['evidence']}")
                if vitro_llm_result.get("evidence"):
                    vitro_evidence_llm_all.append(f"{pmid}: {vitro_llm_result['evidence']}")
        # Set binary flags for in vivo detection
        result["in_vivo_experiment_rule"] = int(bool(vivo_rule_flags))
        result["in_vivo_experiment_mesh"] = int(bool(vivo_mesh_flags))
        if llm_client is None:
            result["in_vivo_experiment_llm"] = ""
        else:
            result["in_vivo_experiment_llm"] = int(bool(vivo_llm_flags))
        final_vivo_flags: List[int] = [
            result["in_vivo_experiment_rule"],
            result["in_vivo_experiment_mesh"],
        ]
        # Only include LLM flag if not empty string
        llm_flag = result["in_vivo_experiment_llm"]
        if isinstance(llm_flag, int):
            final_vivo_flags.append(llm_flag)
        result["in_vivo_experiment_final"] = int(bool(final_vivo_flags and any(final_vivo_flags)))
        result["in_vivo_species_detected"] = ";".join(sorted(set(vivo_species_all)))
        result["in_vivo_evidence_rule"] = "\n".join(vivo_evidence_rule_all)
        result["in_vivo_evidence_mesh"] = "\n".join(vivo_evidence_mesh_all)
        result["in_vivo_evidence_llm"] = "\n".join(vivo_evidence_llm_all)
        # Set binary flags for in vitro detection
        result["in_vitro_experiment_rule"] = int(bool(vitro_rule_flags))
        result["in_vitro_experiment_mesh"] = int(bool(vitro_mesh_flags))
        if llm_client is None:
            result["in_vitro_experiment_llm"] = ""
        else:
            result["in_vitro_experiment_llm"] = int(bool(vitro_llm_flags))
        final_vitro_flags: List[int] = [
            result["in_vitro_experiment_rule"],
            result["in_vitro_experiment_mesh"],
        ]
        llm_flag = result["in_vitro_experiment_llm"]
        if isinstance(llm_flag, int):
            final_vitro_flags.append(llm_flag)
        result["in_vitro_experiment_final"] = int(bool(final_vitro_flags and any(final_vitro_flags)))
        result["in_vitro_cell_types_detected"] = ";".join(sorted(set(vitro_cell_all)))
        result["in_vitro_evidence_rule"] = "\n".join(vitro_evidence_rule_all)
        result["in_vitro_evidence_mesh"] = "\n".join(vitro_evidence_mesh_all)
        result["in_vitro_evidence_llm"] = "\n".join(vitro_evidence_llm_all)
        # Counts and PMIDs
        result["in_vivo_paper_count"] = len(set(vivo_pmid_hits))
        result["in_vitro_paper_count"] = len(set(vitro_pmid_hits))
        result["in_vivo_pmids"] = ";".join(sorted(set(vivo_pmid_hits)))
        result["in_vitro_pmids"] = ";".join(sorted(set(vitro_pmid_hits)))

        result["mutation_sentence"] = "\n".join(mutation_sentence_all)
        result["effect_sentence"] = "\n".join(effect_sentence_all)
        result["methods_sentence"] = "\n".join(methods_sentence_all)
        result["results_sentence"] = "\n".join(results_sentence_all)
        result["discussion_sentence"] = "\n".join(discussion_sentence_all)
        result["host"] = ";".join(sorted(set(host_all)))
        result["cell_line"] = ";".join(sorted(set(cell_line_all)))
        result["experiment_type"] = ";".join(sorted(set(experiment_type_all)))
        result["effect"] = ";".join(sorted(set(effect_all)))
        result["confidence"] = max(confidence_scores) if confidence_scores else 0
        result["effect_terms_found"] = ";".join(sorted(set(effect_terms_all)))
        result["methods_in_vivo"] = int(bool(methods_vivo_flags))
        result["methods_in_vitro"] = int(bool(methods_vitro_flags))
        result["methods_in_vivo_terms_found"] = ";".join(sorted(set(methods_vivo_terms_all)))
        result["methods_in_vitro_terms_found"] = ";".join(sorted(set(methods_vitro_terms_all)))
        result["evidence_grade"] = choose_best_evidence_grade(evidence_grades_all)
    except Exception as exc:
        # Capture any errors
        result["error"] = str(exc)
    return result


def choose_best_evidence_grade(grades: List[str]) -> str:
    """Return the strongest evidence grade among paper-level grades."""
    priority = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    cleaned = [g for g in grades if g in priority]
    if not cleaned:
        return "E"
    return max(cleaned, key=lambda g: priority[g])
