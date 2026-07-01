"""Command‑line entry point for the PubMed mutation search pipeline.

This script ties together the modular components of the pipeline.  It reads
segment/mutation pairs from a tab‑separated file, constructs queries,
executes PubMed searches, applies in vivo and in vitro experiment
classification, aggregates results and writes them to CSV and Excel files.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

from tqdm import tqdm  # type: ignore

from . import config
from .input_parser import parse_input_rows
from .pubmed_client import configure_entrez, get_openai_client
from .result_aggregator import search_and_analyze_mutation
from .output_writer import save_results


def main(argv: List[str] | None = None) -> None:
    """Run the mutation search pipeline for a specified input file."""
    if argv is None:
        argv = sys.argv[1:]
    # Determine input file
    if argv:
        input_file = Path(argv[0])
    else:
        input_file = Path("amino_acid_change.txt")
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    # Read input rows
    try:
        rows = parse_input_rows(input_file)
    except Exception as exc:
        print(f"Error parsing input: {exc}")
        sys.exit(1)
    # Configure Entrez
    configure_entrez()
    # Instantiate LLM client if available
    llm_client = get_openai_client()
    # Prepare results container
    results = []
    fail_count = 0
    # Process each segment/mutation pair
    for segment, mutation in tqdm(rows, desc="PubMed mutation progress", unit="mutation"):
        record = search_and_analyze_mutation(segment, mutation, llm_client=llm_client)
        if record.get("error"):
            fail_count += 1
        results.append(record)
        # Respect the recommended delay between requests
        time.sleep(config.REQUEST_SLEEP_SECONDS)

    # Determine output paths
    csv_path, xlsx_path = config.get_output_paths(input_file)
    # Write results
    save_results(results, csv_path, xlsx_path)
    total = len(results)
    print(f"Processed {total} mutations")
    print(f"Failures: {fail_count}")
    print(f"Results saved to: {csv_path}, {xlsx_path}")


if __name__ == "__main__":
    main()
