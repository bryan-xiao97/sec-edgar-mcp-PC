"""SEC document identity for ``edgar_document``: URLs, attachment selection and readable text.

Split out of ``document.py`` so the tool module stays under the project's
per-file size cap. Everything here is about *which* document a caller means
and whether its text can be read; the tool module owns actions and paging.

``edgar_filing`` also imports ``is_filing_level_filename`` and
``is_hidden_by_default`` so its document hints agree with what
``edgar_document`` will accept.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, NamedTuple, Optional
from urllib.parse import unquote, urlsplit

from edgar.ai.mcp.tools.base import ToolResponse, error

_ACCESSION_DASHED = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_ACCESSION_UNDASHED = re.compile(r"^\d{18}$")
_R_FILE = re.compile(r"^R\d+\.htm$", re.IGNORECASE)
_SEC_ARCHIVE_PATH = re.compile(
    r"/Archives/edgar/data/(?P<cik>\d+)/(?P<accession>\d{18})(?:/(?P<path>[^?#]*))?"
)
# Filenames that name the filing as a whole (index pages, the full submission
# text and its header) rather than one attached document.
_FILING_LEVEL_NAMES = {"index.htm", "index.html", "index.json", "index.xml"}
_FILING_LEVEL_SUFFIXES = ("-index.htm", "-index.html", "-index-headers.html", ".txt", ".hdr.sgml")
# XBRL viewer support files. The XBRL types, R pages and .xml/.xsd files are
# hidden separately below.
_VIEWER_DOCUMENT_TYPES = {"CSS", "JS", "JSON", "ZIP", "XLSX"}
_VIEWER_FILENAMES = {"filingsummary.xml", "metalinks.json", "financial_report.xlsx"}


class SecUrl(NamedTuple):
    """The identity a caller's SEC Archives URL names. The URL is never fetched."""

    accession: str
    cik: int
    filename: Optional[str]


def normalise_accession(value: Optional[str]) -> Optional[str]:
    """Dashed accession for a dashed or 18-digit input, else ``None``."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if _ACCESSION_DASHED.fullmatch(value):
        return value
    if _ACCESSION_UNDASHED.fullmatch(value):
        return f"{value[:10]}-{value[10:12]}-{value[12:]}"
    return None


def is_filing_level_filename(filename: Optional[str], accession_digits: str) -> bool:
    """True when ``filename`` names the filing itself rather than an attachment.

    ``<acc>-index.htm(l)``, ``<acc>-index-headers.html``, ``<acc>.txt`` and
    the folder's ``index.*`` listings are how SEC links a filing; none of them
    is an entry in the filing's attachment list.
    """
    if not filename:
        return True
    lowered = filename.lower()
    if lowered in _FILING_LEVEL_NAMES:
        return True
    dashed = f"{accession_digits[:10]}-{accession_digits[10:12]}-{accession_digits[12:]}"
    for prefix in (dashed, accession_digits):
        if lowered.startswith(prefix) and lowered[len(prefix):] in _FILING_LEVEL_SUFFIXES:
            return True
    return False


def parse_sec_url(value: str) -> tuple[Optional[SecUrl], Optional[str]]:
    """Validate a caller URL and return its identity or an error message.

    Only ``https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/[file]``
    is accepted. A filing-level filename (index page, full submission) is
    reported as ``filename=None``; a nested subdirectory path is rejected.
    """
    if not isinstance(value, str) or not value:
        return None, "Provide an HTTPS SEC Archives URL."
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        return None, f"The URL could not be parsed: {exc}"
    if parsed.scheme != "https" or parsed.netloc != "www.sec.gov":
        return None, "Only HTTPS URLs with the exact authority www.sec.gov are accepted."
    if not parsed.path.startswith("/Archives/"):
        return None, "The SEC URL path must begin with /Archives/."

    match = _SEC_ARCHIVE_PATH.fullmatch(parsed.path)
    if not match:
        return None, "The URL must identify an SEC filing under /Archives/edgar/data/ with an accession number."

    accession_digits = match.group("accession")
    filename = None
    path = match.group("path") or ""
    if path:
        try:
            decoded = unquote(path, errors="strict")
        except (UnicodeDecodeError, ValueError):
            return None, "The URL contains an invalid encoded path."
        if "/" in decoded or "\\" in decoded or decoded in {".", ".."}:
            return None, "The URL must name a document directly inside the filing folder, not a subdirectory."
        filename = None if is_filing_level_filename(decoded, accession_digits) else decoded

    accession = f"{accession_digits[:10]}-{accession_digits[10:12]}-{accession_digits[12:]}"
    return SecUrl(accession=accession, cik=int(match.group("cik")), filename=filename), None


def url_cik_matches(filing, url_cik: int) -> bool:
    """Whether the URL's CIK segment is the filing's filer or a related filer."""
    try:
        if int(filing.cik) == url_cik:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return url_cik in {int(cik) for cik in filing.all_ciks}
    except Exception:
        return False


def filer_of(filing) -> dict:
    return {"cik": int(filing.cik), "name": getattr(filing, "company", None)}


def period_of_report(filing) -> Optional[str]:
    """``report_date`` when present, else the SGML header's period once the SGML is loaded.

    ``find(accession)`` returns a plain ``Filing`` without ``report_date``.
    Listing attachments has already loaded the SGML, so its header period is
    free; before that (``_sgml`` unset) nothing is fetched just for provenance.
    """
    report_date = getattr(filing, "report_date", None)
    if report_date:
        return str(report_date)
    if getattr(filing, "_sgml", None) is None:
        return None
    try:
        period = filing.period_of_report
    except Exception:
        return None
    return str(period) if period else None


def filing_source(filing, selected_by: str, selected_attachment=None) -> dict:
    """The provenance ``source`` block (constraints rule 3)."""
    source = {
        "cik": int(filing.cik),
        "entity": getattr(filing, "company", None),
        "accession_number": filing.accession_number,
        "form": filing.form,
        "period_of_report": period_of_report(filing),
        "filed": str(filing.filing_date),
        "url": filing.homepage_url,
        "is_amendment": str(filing.form).endswith("/A"),
        "selected_by": selected_by,
    }
    if selected_attachment is not None:
        source["document_url"] = safe_url(selected_attachment)
    return source


def safe_url(attachment) -> Optional[str]:
    try:
        return getattr(attachment, "url", None)
    except Exception:
        return None


def identity(attachment) -> dict[str, Any]:
    sequence = str(getattr(attachment, "sequence_number", "") or "")
    primary = sequence == "1"
    doc_type = getattr(attachment, "document_type", None)
    return {
        "sequence": sequence,
        "sequence_number": sequence,
        "filename": getattr(attachment, "document", None),
        "type": doc_type,
        "document_type": doc_type,
        "description": getattr(attachment, "display_description", None)
        or getattr(attachment, "description", None),
        "size": getattr(attachment, "size", None),
        "primary": primary,
        "is_primary": primary,
        "readable": readable_format(attachment),
        "url": safe_url(attachment),
    }


def readable_format(attachment) -> bool:
    if is_paper_attachment(attachment):
        return False
    if getattr(attachment, "empty", False) or getattr(attachment, "is_binary", lambda: False)():
        return False
    return bool(getattr(attachment, "is_text", lambda: False)())


def is_paper_attachment(attachment) -> bool:
    filename = str(getattr(attachment, "document", "") or "")
    return PurePosixPath(filename).suffix.lower() == ".paper"


def is_hidden_by_default(attachment) -> bool:
    """XBRL files, generated R pages and XBRL viewer support files."""
    # The filing's sequence-1 document is the main report, even when it is
    # inline XBRL. Keep it available while hiding separate XBRL support files.
    if str(getattr(attachment, "sequence_number", "")) == "1":
        return False
    filename = str(getattr(attachment, "document", "") or "")
    lowered = filename.lower()
    doc_type = str(getattr(attachment, "document_type", "") or "").upper()
    extension = PurePosixPath(filename).suffix.lower()
    return bool(
        _R_FILE.fullmatch(filename)
        or getattr(attachment, "ixbrl", False)
        or extension in {".xml", ".xbrl", ".xsd"}
        or doc_type.startswith("EX-101")
        or doc_type in {"XML", "XBRL", "EX-104"}
        or doc_type in _VIEWER_DOCUMENT_TYPES
        or lowered in _VIEWER_FILENAMES
        or lowered.endswith("-xbrl.zip")
    )


def eligible(attachments: list, include_all: bool) -> list:
    return attachments if include_all else [a for a in attachments if not is_hidden_by_default(a)]


def list_attachments(filing) -> list:
    try:
        return list(filing.attachments)
    except Exception as exc:
        raise RuntimeError(f"Could not list filing attachments: {exc}") from exc


def _type_matches(attachments: list, selector: str) -> list:
    normalized = selector.upper()
    if re.fullmatch(r"EX-\d+", normalized):
        # A bare exhibit number is a prefix, so EX-10 deliberately covers
        # EX-10.1, EX-10.2 and similar variants. Ambiguity is surfaced later.
        return [
            a for a in attachments
            if (doc_type := str(getattr(a, "document_type", "") or "").upper()) == normalized
            or doc_type.startswith(normalized + ".")
        ]
    if normalized.startswith("EX-"):
        # A qualified exhibit type (EX-10.1) is an exact identity. Treating it
        # as a raw prefix would also select EX-10.10 when EX-10.1 is absent.
        return [a for a in attachments if str(getattr(a, "document_type", "") or "").upper() == normalized]
    return [a for a in attachments if str(getattr(a, "document_type", "") or "").upper().startswith(normalized)]


def _selector_matches(attachments: list, selector: str) -> list:
    if selector.isdigit():
        return [a for a in attachments if str(getattr(a, "sequence_number", "")) == selector]
    exact_filename = [a for a in attachments if getattr(a, "document", None) == selector]
    if exact_filename:
        return exact_filename
    if selector.lower() == "primary":
        return [a for a in attachments if str(getattr(a, "sequence_number", "")) == "1"]
    return _type_matches(attachments, selector)


def _single_match(matches: list, selector: str, filer: Optional[dict]) -> tuple[Optional[Any], Optional[ToolResponse]]:
    if not matches:
        return None, error(
            f"No filing attachment matches document selector {selector!r}.",
            suggestions=["List this filing's documents and use an exact sequence or filename."],
            error_code="DOCUMENT_NOT_FOUND",
        )
    if len(matches) > 1:
        data: dict[str, Any] = {"candidates": [identity(a) for a in matches]}
        if filer is not None:
            data["filer"] = filer
        return None, ToolResponse(
            success=False,
            error=f"Document selector {selector!r} matches multiple attachments.",
            error_code="AMBIGUOUS_DOCUMENT",
            suggestions=["Choose one candidate by exact filename or sequence number."],
            data=data,
        )
    return matches[0], None


def resolve_attachment(
    attachments: list, selector: Optional[str], filer: Optional[dict] = None
) -> tuple[Optional[Any], Optional[ToolResponse]]:
    """One attachment by sequence, exact filename, ``primary`` or exhibit type."""
    if selector is None or not str(selector).strip():
        return None, error(
            "A document selector is required for this action.",
            suggestions=["Use a sequence number, exact filename, or exhibit type such as EX-32.1."],
            error_code="DOCUMENT_REQUIRED",
        )
    selector = str(selector).strip()
    return _single_match(_selector_matches(attachments, selector), selector, filer)


def resolve_url_filename(
    attachments: list, filename: str, filer: Optional[dict] = None
) -> tuple[Optional[Any], Optional[ToolResponse]]:
    """A URL's filename binds by exact ``attachment.document`` only (never type or sequence)."""
    matches = [a for a in attachments if getattr(a, "document", None) == filename]
    return _single_match(matches, filename, filer)


def library_version() -> str:
    try:
        import edgar
        return str(edgar.__version__)
    except Exception:
        return "unknown"


def unreadable_reason(attachment, render_error: Optional[str] = None) -> str:
    if getattr(attachment, "empty", False):
        return "The filing attachment has no document filename or content."
    if is_paper_attachment(attachment):
        return "Paper filing attachments are not rendered as readable document text by this tool."
    if getattr(attachment, "is_binary", lambda: False)():
        extension = PurePosixPath(str(getattr(attachment, "document", ""))).suffix.lower()
        if extension == ".pdf":
            return "This PDF attachment has no extracted text available through the filing text renderer."
        kind = extension.lstrip(".") or "binary"
        return f"This {kind} attachment has no extracted text available through the filing text renderer."
    if render_error:
        return f"The attachment could not be rendered as text: {render_error}"
    return "The attachment did not produce readable text."


def _render(attachment) -> tuple[Optional[str], Optional[str]]:
    """Rendered markdown, falling back to plain text, plus the last render error."""
    render_error = None
    text = None
    try:
        text = attachment.markdown()
    except Exception as exc:
        render_error = str(exc)
    if not text:
        try:
            text = attachment.text()
        except Exception as exc:
            render_error = str(exc)
    if text is None:
        return None, render_error
    return (text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)), render_error


def read_text(filing, attachment) -> tuple[Optional[str], Optional[str]]:
    """Full rendered text or an unreadable reason, using the shared text cache.

    Read and search both go through here, so a search locator's
    ``char_offset`` always indexes the same text that ``read`` pages.
    """
    from edgar.ai.mcp.tools.continuation import text_cache

    if is_paper_attachment(attachment) or getattr(attachment, "empty", False):
        return None, unreadable_reason(attachment)

    filename = str(getattr(attachment, "document", "") or "")
    cache_key = ("edgar_document", filing.accession_number, filename, library_version())
    cached = text_cache.get(cache_key)
    if cached is not None:
        return cached, None

    if getattr(attachment, "is_binary", lambda: False)():
        return None, unreadable_reason(attachment)

    text, render_error = _render(attachment)
    if text is None or not text.strip():
        return None, unreadable_reason(attachment, render_error)

    text_cache.put(cache_key, text, size_bytes=len(text.encode("utf-8")))
    return text, None
