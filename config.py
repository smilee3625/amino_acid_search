"""Configuration and shared constants for the PubMed mutation search pipeline.

This module centralises values that are used across the pipeline, such as
Entrez settings, model configuration for the optional large language model,
and the default query template.  Keeping these definitions in one place
facilitates maintenance and ensures consistent values are used throughout
the codebase.
"""

from pathlib import Path
from datetime import datetime

# Third‑party imports are kept local to modules where they are used.  The
# configuration module defines only simple constants and does not trigger
# heavy imports.

###############################################################################
# Entrez configuration
###############################################################################

# Your email address is required by NCBI when using Entrez.  Replace this
# value with your actual email address to comply with NCBI guidelines.
ENTREZ_EMAIL: str = "smilee3625@naver.com"

# Optional tool name sent to Entrez.  Set to a string such as
# "AI_mutation_search" if you wish to provide a custom identifier.
ENTREZ_TOOL: str | None = None

###############################################################################
# LLM configuration
###############################################################################

# Toggle use of the OpenAI client.  When set to ``False`` the pipeline
# performs only rule‑ and MeSH‑based experiment detection.
USE_LLM: bool = True

# Name of the model to use when invoking the OpenAI API.  This value is
# ignored unless ``USE_LLM`` is true and a valid API key is present in
# ``OPENAI_API_KEY``.
LLM_MODEL: str = "gpt-5.5"
###############################################################################
# Open-access full-text configuration
###############################################################################

# Use PMC full-text XML when a PubMed record has a linked PMCID.
USE_PMC_FULLTEXT: bool = True

# Use Europe PMC as an additional open-access full-text source.
# This is prepared for pipeline expansion beyond PMC.
USE_EUROPEPMC: bool = True

# Use Unpaywall as a fallback source when PMC/Europe PMC full text is unavailable.
USE_UNPAYWALL: bool = True
UNPAYWALL_EMAIL: str = "smilee3625@naver.com"

# Timeout in seconds for each full-text retrieval request.
# This prevents the pipeline from hanging for several minutes on Entrez.elink or OA requests.
FULLTEXT_TIMEOUT: int = 15
USE_CROSSREF: bool = True
USE_OPENALEX: bool = True
USE_SEMANTIC_SCHOLAR: bool = True
USE_PUBLISHER_DIRECT: bool = True

CROSSREF_MAILTO: str = "smilee3625@naver.com"
SEMANTIC_SCHOLAR_API_KEY: str = ""

###############################################################################
# PubMed search configuration
###############################################################################

# Maximum number of PMIDs to fetch detailed records for when analysing each
# mutation.  Setting a modest upper bound helps avoid excessive API calls
# while still capturing a representative sample of the literature.
MAX_PMIDS_FOR_DETAIL_CHECK: int = 50

# Default pause (in seconds) between PubMed queries.  NCBI recommends
# introducing a delay between successive requests to avoid overloading
# their servers.  The delay can be adjusted dynamically in the main
# pipeline as needed.
REQUEST_SLEEP_SECONDS: float = 0.4

# Template for constructing a PubMed query.  The placeholders ``{segment}``
# and ``{mutation}`` will be substituted with the actual values when the
# query is built.  This template restricts searches to avian influenza
# literature. Review articles are not excluded here; they should be detected
# and labelled during downstream analysis.
QUERY_TEMPLATE: str = (
    '("Influenza in Birds"[MeSH Terms] OR '
    '"avian influenza"[Title/Abstract] OR '
    '"avian influenza virus"[Title/Abstract] OR '
    '"highly pathogenic avian influenza"[Title/Abstract] OR '
    'HPAI[Title/Abstract] OR LPAI[Title/Abstract] OR '
    'H5N1[Title/Abstract] OR H7N9[Title/Abstract] OR H5N6[Title/Abstract] OR '
    'H5N8[Title/Abstract] OR H9N2[Title/Abstract]) '
    'AND {segment}[Title/Abstract] '
    'AND {mutation}[All Fields]'
)

###############################################################################
# Output configuration
###############################################################################

# Generate timestamped output filenames based on the current system time.  The
# timestamp helps prevent overwriting previous runs and makes it easy to
# distinguish multiple analysis runs.  These variables will be used by
# ``run_pipeline.py`` when writing CSV and Excel files.
timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_output_paths(input_file: Path) -> tuple[Path, Path]:
    """Return the output CSV and XLSX paths for a given input file.

    Parameters
    ----------
    input_file:
        Path to the input text file describing segment/mutation pairs.

    Returns
    -------
    (csv_path, xlsx_path):
        Tuple containing the paths to the CSV and Excel outputs.  The
        filenames incorporate the stem of the input file and the current
        timestamp.
    """
    input_stem = input_file.stem
    csv_path = Path(f"pubmed_mutation_results_{input_stem}_{timestamp}.csv")
    xlsx_path = Path(f"pubmed_mutation_results_{input_stem}_{timestamp}.xlsx")
    return csv_path, xlsx_path

###############################################################################
# Result column definitions
###############################################################################

# Columns to be included in the final results DataFrame.  These names must
# align with the keys returned by the search and analysis functions.
RESULT_COLUMNS: list[str] = [
    "segment",
    "mutation",
    "query",
    "query_count",
    "analyzed_paper_count",
    "in_vivo_paper_count",
    "in_vitro_paper_count",
    "review_paper_count",
    "review_pmid_list",
    "in_vivo_pmids",
    "in_vitro_pmids",
    "full_text_available",
    "full_text_paper_count",
    "full_text_pmids",
    "full_text_source",
    "full_text_status",
    "no_full_text_paper_count",
    "no_full_text_pmids",
    "in_vivo_pmids",
    "in_vivo_experiment_rule",
    "in_vivo_experiment_mesh",
    "in_vivo_experiment_llm",
    "in_vivo_experiment_final",
    "in_vivo_species_detected",
    "in_vivo_evidence_rule",
    "in_vivo_evidence_mesh",
    "in_vivo_evidence_llm",
    "in_vitro_experiment_rule",
    "in_vitro_experiment_mesh",
    "in_vitro_experiment_llm",
    "in_vitro_experiment_final",
    "in_vitro_cell_types_detected",
    "in_vitro_evidence_rule",
    "in_vitro_evidence_mesh",
    "in_vitro_evidence_llm",
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
    "error",
]
