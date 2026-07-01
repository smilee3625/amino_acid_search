"""PubMed interaction utilities for the mutation search pipeline.

This module wraps the NCBI Entrez API calls and provides helper functions to
execute searches and fetch detailed records.  It also offers a convenience
function for instantiating the OpenAI client, when enabled.
"""

from __future__ import annotations

import os
import socket
import tempfile
import time
import requests
from typing import Any, Callable, Iterable, List, Tuple
from xml.etree import ElementTree
from Bio import Medline  # type: ignore
from Bio import Entrez  # type: ignore
from urllib.parse import urljoin

from . import config

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 PubMedMutationPipeline/1.0"
}

MIN_EXTRACTED_TEXT_LENGTH = 200

ENTREZ_DEFAULT_TIMEOUT = 20
ENTREZ_DEFAULT_MAX_RETRIES = 2
def get_with_retries(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> requests.Response | None:
    """GET request wrapper with timeout, retries, and a consistent User-Agent."""
    request_timeout = timeout if timeout is not None else getattr(config, "FULLTEXT_TIMEOUT", 15)
    retries = max_retries if max_retries is not None else getattr(config, "FULLTEXT_MAX_RETRIES", 2)
    merged_headers = dict(REQUEST_HEADERS)
    if headers:
        merged_headers.update(headers)

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=merged_headers,
                timeout=request_timeout,
            )
            if response.status_code == 200:
                return response
            if response.status_code in {400, 401, 403, 404}:
                return response
        except requests.RequestException:
            if attempt >= retries:
                return None

    return None


def is_usable_text(text: str, min_length: int = MIN_EXTRACTED_TEXT_LENGTH) -> bool:
    """Return True when extracted text is long enough to be useful."""
    return bool(text and len(text.strip()) >= min_length)


def extract_doi_from_medline_record(record: dict[str, object]) -> str:
    """Extract DOI from a MEDLINE record when available."""
    for aid in record.get("AID", []) or []:  # type: ignore[union-attr]
        aid_text = str(aid)
        if "[doi]" in aid_text.lower():
            return aid_text.replace("[doi]", "").strip()
    return ""


def configure_entrez() -> None:
    """Apply configuration settings to the Entrez module.

    This function should be called once at the start of the pipeline to
    ensure that NCBI's requirements (such as specifying an email address)
    are satisfied.  The optional tool name is also applied if provided.
    """
    Entrez.email = config.ENTREZ_EMAIL
    if config.ENTREZ_TOOL is not None:
        Entrez.tool = config.ENTREZ_TOOL


def get_entrez_timeout() -> int:
    """Return Entrez timeout in seconds."""
    return int(getattr(config, "ENTREZ_TIMEOUT", getattr(config, "FULLTEXT_TIMEOUT", ENTREZ_DEFAULT_TIMEOUT)))


def get_entrez_max_retries() -> int:
    """Return maximum Entrez retry count."""
    return int(getattr(config, "ENTREZ_MAX_RETRIES", ENTREZ_DEFAULT_MAX_RETRIES))


def safe_entrez_read(
    handle_factory: Callable[[], Any],
    *,
    parser: str = "entrez",
    default: Any = None,
) -> Any:
    """Read Entrez results with timeout and retry.

    Parameters
    ----------
    handle_factory:
        Function that creates an Entrez handle, for example
        ``lambda: Entrez.esearch(...)`` or ``lambda: Entrez.efetch(...)``.

    parser:
        ``"entrez"`` uses ``Entrez.read``. ``"medline"`` uses
        ``Medline.parse`` and returns a list.

    default:
        Value returned when all attempts fail.
    """
    timeout = get_entrez_timeout()
    retries = get_entrez_max_retries()
    previous_timeout = socket.getdefaulttimeout()

    try:
        socket.setdefaulttimeout(timeout)

        for attempt in range(retries + 1):
            try:
                with handle_factory() as handle:
                    if parser == "medline":
                        return list(Medline.parse(handle))
                    return Entrez.read(handle)
            except Exception:
                if attempt >= retries:
                    return default
                time.sleep(min(2 * (attempt + 1), 5))

        return default

    finally:
        socket.setdefaulttimeout(previous_timeout)


def run_pubmed_esearch(query: str) -> Tuple[int, List[str]]:
    """Execute a PubMed search query and return hit count and PMID list.

    Parameters
    ----------
    query:
        PubMed search query string.

    Returns
    -------
    count:
        Total number of records matching the query.

    id_list:
        List of PMIDs returned by the search (limited by the retmax parameter).
    """
    record = safe_entrez_read(
        lambda: Entrez.esearch(db="pubmed", term=query, retmax=10000),
        parser="entrez",
        default={},
    )
    count_str = record.get("Count", "0")
    id_list = record.get("IdList", [])
    try:
        count = int(count_str)
    except ValueError:
        count = 0
    return count, id_list


def fetch_pubmed_details(pmid_list: Iterable[str]) -> list[dict[str, object]]:
    """Fetch MEDLINE details for a list of PMIDs.

    Parameters
    ----------
    pmid_list:
        Iterable of PubMed identifiers (PMIDs) to fetch.

    Returns
    -------
    list of dict:
        Each dictionary contains keys such as ``PMID``, ``title``,
        ``abstract``, ``mesh_terms`` (list of MeSH terms) and
        ``publication_types``.  If an exception occurs during fetching,
        an empty list is returned.
    """
    pmids = list(pmid_list)
    if not pmids:
        return []
    records: list[dict[str, object]] = []
    medline_records = safe_entrez_read(
        lambda: Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="medline",
            retmode="text",
        ),
        parser="medline",
        default=[],
    )

    for rec in medline_records:
        records.append({
            "PMID": rec.get("PMID", ""),
            "title": rec.get("TI", ""),
            "abstract": rec.get("AB", ""),
            "journal": rec.get("JT", ""),
            "year": str(rec.get("DP", ""))[:4],
            "doi": extract_doi_from_medline_record(rec),
            "mesh_terms": rec.get("MH", []),
            "publication_types": rec.get("PT", []),
        })

    return records


def fetch_pmc_fulltext_from_pmid(pmid: str) -> str:
    """Fetch PMC full-text XML for a PMID if available.

    This implementation uses NCBI E-utilities through `requests` with a
    timeout. This prevents the pipeline from hanging indefinitely at
    Entrez.elink when NCBI response is delayed.
    """
    if not pmid or not getattr(config, "USE_PMC_FULLTEXT", True):
        return ""

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)

    try:
        link_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        link_params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "xml",
            "email": config.ENTREZ_EMAIL,
        }
        if config.ENTREZ_TOOL is not None:
            link_params["tool"] = config.ENTREZ_TOOL

        link_response = get_with_retries(link_url, params=link_params, timeout=timeout)
        if link_response is None or link_response.status_code != 200:
            return ""

        root = ElementTree.fromstring(link_response.text)
        pmc_ids = [elem.text for elem in root.findall(".//Link/Id") if elem.text]

        if not pmc_ids:
            return ""

        pmc_id = pmc_ids[0]

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "full",
            "retmode": "xml",
            "email": config.ENTREZ_EMAIL,
        }
        if config.ENTREZ_TOOL is not None:
            fetch_params["tool"] = config.ENTREZ_TOOL

        fetch_response = get_with_retries(fetch_url, params=fetch_params, timeout=timeout)
        if fetch_response is None or fetch_response.status_code != 200:
            return ""

        return fetch_response.text

    except Exception:
        return ""

def extract_text_from_pmc_xml(xml_text: str) -> str:
    """Extract readable text from PMC XML, including table cell text."""
    if not xml_text:
        return ""

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)
        texts: list[str] = []

        for elem in root.iter():
            if elem.text:
                texts.append(elem.text)

            # table, inline tag, nested tag 뒤쪽 텍스트까지 수집
            if elem.tail:
                texts.append(elem.tail)

        return " ".join(texts)

    except Exception:
        return ""

def fetch_europepmc_fulltext_from_pmid(pmid: str) -> str:
    """Fetch full-text XML from Europe PMC for a PMID if available."""
    if not pmid or not getattr(config, "USE_EUROPEPMC", False):
        return ""

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)

    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmid}/fullTextXML"
        response = get_with_retries(url, timeout=timeout)

        if response is None or response.status_code != 200:
            return ""

        text = response.text or ""
        if "<article" not in text and "<body" not in text:
            return ""

        return text

    except Exception:
        return ""

def is_review_article(record: dict[str, object]) -> bool:
    """Determine if a MEDLINE record is a review article.

    Parameters
    ----------
    record:
        Dictionary representing a MEDLINE record as returned by
        ``fetch_pubmed_details``.

    Returns
    -------
    bool:
        ``True`` if the ``publication_types`` field contains the word
        "review" (case‑insensitive), otherwise ``False``.
    """
    pt = [p.lower() for p in record.get("publication_types", [])]  # type: ignore
    return "review" in pt


def get_openai_client():
    """Return an OpenAI client if the API key and package are available.

    The optional large language model (LLM) is used to classify PubMed
    records when rule‑based and MeSH‑based detectors do not identify an
    experiment.  If ``USE_LLM`` is false, or the environment variable
    ``OPENAI_API_KEY`` is not set, or the ``openai`` package is not
    installed, this function returns ``None``.

    Returns
    -------
    OpenAI | None:
        An instantiated OpenAI client if available; otherwise ``None``.
    """
    if not config.USE_LLM:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None
    return OpenAI(api_key=api_key)

def fetch_doi_from_pmid(pmid: str) -> str | None:
    """Fetch DOI for a PMID using NCBI E-utilities XML with timeout."""
    if not pmid:
        return None

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)

    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "email": config.ENTREZ_EMAIL,
        }
        if config.ENTREZ_TOOL is not None:
            params["tool"] = config.ENTREZ_TOOL

        response = get_with_retries(url, params=params, timeout=timeout)
        if response is None or response.status_code != 200:
            return None

        root = ElementTree.fromstring(response.text)
        for article_id in root.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi" and article_id.text:
                return article_id.text.strip()

    except Exception:
        return None

    return None



# --- Unpaywall fulltext utilities ---

def extract_text_from_html(html_text: str) -> str:
    """Extract readable text from an HTML full-text page."""
    if not html_text:
        return ""

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html_text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)

    except Exception:
        return ""


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract readable text from PDF bytes using pdfminer.six."""
    if not pdf_bytes:
        return ""

    try:
        from pdfminer.high_level import extract_text  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()
            return extract_text(temp_pdf.name) or ""

    except Exception:
        return ""

def extract_text_from_url(url: str) -> str:
    """Fetch PDF or HTML URL and extract readable text."""
    if not url:
        return ""

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)

    try:
        response = get_with_retries(url, timeout=timeout)

        if response is None or response.status_code != 200:
            return ""

        content_type = response.headers.get("content-type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return extract_text_from_pdf_bytes(response.content)

        if "html" in content_type or "text" in content_type:
            return extract_text_from_html(response.text)

        return ""

    except Exception:
        return ""


def find_pdf_links_from_html_url(url: str) -> list[str]:
    """Find PDF links from a publisher landing page."""
    if not url:
        return []

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)

    try:
        response = get_with_retries(url, timeout=timeout)

        if response is None or response.status_code != 200:
            return []

        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(" ", strip=True).lower()

            if ".pdf" in href.lower() or "pdf" in text:
                pdf_links.append(urljoin(url, href))

        return list(dict.fromkeys(pdf_links))

    except Exception:
        return []


def fetch_crossref_fulltext_from_doi(doi: str) -> tuple[str, str]:
    """Try Crossref full-text links and publisher landing page."""
    if not doi or not getattr(config, "USE_CROSSREF", False):
        return "", ""

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)
    mailto = getattr(config, "CROSSREF_MAILTO", "")

    try:
        api_url = f"https://api.crossref.org/works/{doi}"
        params = {}
        if mailto:
            params["mailto"] = mailto

        response = get_with_retries(api_url, params=params, timeout=timeout)

        if response is None or response.status_code != 200:
            return "", ""

        data = response.json()
        message = data.get("message", {})

        # 1. Crossref metadata의 full-text link 시도
        for link in message.get("link", []) or []:
            url = link.get("URL", "")
            if not url:
                continue

            text = extract_text_from_url(url)
            if text.strip():
                return text, "CROSSREF_LINK"

        # 2. DOI landing page 시도
        landing_url = message.get("URL", "")
        if landing_url:
            text = extract_text_from_url(landing_url)
            if text.strip():
                return text, "CROSSREF_LANDING_HTML"

            for pdf_url in find_pdf_links_from_html_url(landing_url):
                pdf_text = extract_text_from_url(pdf_url)
                if pdf_text.strip():
                    return pdf_text, "CROSSREF_LANDING_PDF"

        return "", ""

    except Exception:
        return "", ""


def fetch_openalex_fulltext(pmid: str = "", doi: str = "") -> tuple[str, str]:
    """Try OpenAlex OA URL by DOI first, then PMID."""
    if not getattr(config, "USE_OPENALEX", False):
        return "", ""

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)

    identifiers = []
    if doi:
        identifiers.append(f"doi:{doi}")
    if pmid:
        identifiers.append(f"pmid:{pmid}")

    for identifier in identifiers:
        try:
            api_url = f"https://api.openalex.org/works/{identifier}"
            response = get_with_retries(api_url, timeout=timeout)

            if response is None or response.status_code != 200:
                continue

            data = response.json()

            urls = []

            open_access = data.get("open_access", {}) or {}
            if open_access.get("oa_url"):
                urls.append(open_access["oa_url"])

            primary_location = data.get("primary_location", {}) or {}
            if primary_location.get("pdf_url"):
                urls.append(primary_location["pdf_url"])
            if primary_location.get("landing_page_url"):
                urls.append(primary_location["landing_page_url"])

            for loc in data.get("locations", []) or []:
                if loc.get("pdf_url"):
                    urls.append(loc["pdf_url"])
                if loc.get("landing_page_url"):
                    urls.append(loc["landing_page_url"])

            for url in list(dict.fromkeys(urls)):
                text = extract_text_from_url(url)
                if text.strip():
                    return text, "OPENALEX"

                for pdf_url in find_pdf_links_from_html_url(url):
                    pdf_text = extract_text_from_url(pdf_url)
                    if pdf_text.strip():
                        return pdf_text, "OPENALEX_PDF"

        except Exception:
            continue

    return "", ""


def fetch_semantic_scholar_fulltext(doi: str) -> tuple[str, str]:
    """Try Semantic Scholar openAccessPdf by DOI."""
    if not doi or not getattr(config, "USE_SEMANTIC_SCHOLAR", False):
        return "", ""

    timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)

    try:
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {
            "fields": "title,openAccessPdf,externalIds,url"
        }

        headers = {}
        api_key = getattr(config, "SEMANTIC_SCHOLAR_API_KEY", "")
        if api_key:
            headers["x-api-key"] = api_key

        response = get_with_retries(
            api_url,
            params=params,
            headers=headers,
            timeout=timeout,
        )

        if response is None or response.status_code != 200:
            return "", ""

        data = response.json()
        oa_pdf = data.get("openAccessPdf") or {}

        pdf_url = oa_pdf.get("url", "")
        if pdf_url:
            text = extract_text_from_url(pdf_url)
            if text.strip():
                return text, "SEMANTIC_SCHOLAR"

        return "", ""

    except Exception:
        return "", ""


def fetch_publisher_direct_fulltext(doi: str) -> tuple[str, str]:
    """Try publisher landing page and PDF links through DOI resolver."""
    if not doi or not getattr(config, "USE_PUBLISHER_DIRECT", False):
        return "", ""

    landing_url = f"https://doi.org/{doi}"

    try:
        text = extract_text_from_url(landing_url)
        if text.strip():
            return text, "PUBLISHER_HTML"

        for pdf_url in find_pdf_links_from_html_url(landing_url):
            pdf_text = extract_text_from_url(pdf_url)
            if pdf_text.strip():
                return pdf_text, "PUBLISHER_PDF"

        return "", ""

    except Exception:
        return "", ""


def fetch_unpaywall_fulltext_from_doi(doi: str) -> str:
    """Fetch OA full text through Unpaywall using DOI.

    This function asks Unpaywall for OA locations. It first tries PDF URLs,
    then HTML landing-page URLs. It returns extracted text, not raw HTML/PDF.
    If no usable OA text is found, it returns an empty string.
    """
    if not doi:
        return ""

    if not getattr(config, "USE_UNPAYWALL", False):
        return ""

    email = getattr(config, "UNPAYWALL_EMAIL", "")
    if not email:
        return ""

    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

    try:
        timeout = getattr(config, "FULLTEXT_TIMEOUT", 15)
        response = get_with_retries(api_url, timeout=timeout)
        if response is None or response.status_code != 200:
            return ""

        data = response.json()
        oa_locations = data.get("oa_locations") or []

        for location in oa_locations:
            pdf_url = location.get("url_for_pdf")
            if not pdf_url:
                continue

            try:
                pdf_response = get_with_retries(pdf_url, timeout=timeout)
                if pdf_response is None:
                    continue
                content_type = pdf_response.headers.get("content-type", "").lower()

                if pdf_response.status_code == 200 and (
                    "pdf" in content_type or pdf_url.lower().endswith(".pdf")
                ):
                    pdf_text = extract_text_from_pdf_bytes(pdf_response.content)
                    if pdf_text.strip():
                        return pdf_text

            except Exception:
                continue

        for location in oa_locations:
            html_url = location.get("url") or location.get("url_for_landing_page")
            if not html_url:
                continue

            try:
                html_response = get_with_retries(html_url, timeout=timeout)
                if html_response is None:
                    continue
                content_type = html_response.headers.get("content-type", "").lower()

                if html_response.status_code == 200 and "html" in content_type:
                    html_text = extract_text_from_html(html_response.text)
                    if html_text.strip():
                        return html_text

            except Exception:
                continue

        return ""

    except Exception:
        return ""


# Backward-compatible alias for older result_aggregator.py imports.
def fetch_oa_fulltext_from_doi(doi: str) -> str:
    """Alias for fetch_unpaywall_fulltext_from_doi."""
    return fetch_unpaywall_fulltext_from_doi(doi)


def get_best_fulltext(pmid: str) -> tuple[str, str, str]:
    """Fetch the best available open-access full text for a PMID.

    Priority:
    1. PMC XML
    2. Europe PMC XML
    3. DOI
    4. Crossref
    5. Unpaywall
    6. OpenAlex
    7. Semantic Scholar
    8. Publisher HTML/PDF

    Returns:
        full_text, source, status
    """
    if not pmid:
        return "", "", "NOT_FOUND"

    try:
        # 1. PMC
        pmc_xml = fetch_pmc_fulltext_from_pmid(pmid)
        pmc_text = extract_text_from_pmc_xml(pmc_xml)
        if is_usable_text(pmc_text):
            return pmc_text, "PMC", "SUCCESS"

        # 2. Europe PMC
        europepmc_xml = fetch_europepmc_fulltext_from_pmid(pmid)
        europepmc_text = extract_text_from_pmc_xml(europepmc_xml)
        if is_usable_text(europepmc_text):
            return europepmc_text, "EUROPEPMC", "SUCCESS"

        # 3. DOI
        doi = fetch_doi_from_pmid(pmid)

        if doi:
            # 4. Crossref
            crossref_text, crossref_source = fetch_crossref_fulltext_from_doi(doi)
            if is_usable_text(crossref_text):
                return crossref_text, crossref_source, "SUCCESS"

            # 5. Unpaywall
            unpaywall_text = fetch_unpaywall_fulltext_from_doi(doi)
            if is_usable_text(unpaywall_text):
                return unpaywall_text, "UNPAYWALL", "SUCCESS"

            # 6. OpenAlex
            openalex_text, openalex_source = fetch_openalex_fulltext(
                pmid=pmid,
                doi=doi,
            )
            if is_usable_text(openalex_text):
                return openalex_text, openalex_source, "SUCCESS"

            # 7. Semantic Scholar
            sem_text, sem_source = fetch_semantic_scholar_fulltext(doi)
            if is_usable_text(sem_text):
                return sem_text, sem_source, "SUCCESS"

            # 8. Publisher direct
            publisher_text, publisher_source = fetch_publisher_direct_fulltext(doi)
            if is_usable_text(publisher_text):
                return publisher_text, publisher_source, "SUCCESS"

        else:
            # DOI가 없어도 OpenAlex는 PMID로 조회 가능
            openalex_text, openalex_source = fetch_openalex_fulltext(
                pmid=pmid,
                doi="",
            )
            if is_usable_text(openalex_text):
                return openalex_text, openalex_source, "SUCCESS"

        return "", "", "NOT_FOUND"

    except Exception:
        return "", "", "ERROR"
