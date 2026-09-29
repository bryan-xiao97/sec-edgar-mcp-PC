"""Bounded, honest attachment search for ``edgar_document``.

Search reads every document through ``document_identity.read_text`` -- the
same cached text ``read`` pages -- so a locator's ``char_offset`` always
indexes the text a follow-up read returns, and a document whose text cannot
be read is reported in ``unsearched_documents`` rather than silently counted
as "no matches" (QA finding P2-H1).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from edgar.ai.mcp.tools.continuation import fingerprint
from edgar.ai.mcp.tools.document_identity import library_version, read_text
from edgar.search.grep import _grep_text

MAX_SEARCH_MATCHES = 1_000
MAX_SEARCH_MATCH_CHARS = 2_048
REGEX_TIMEOUT_SECONDS = 0.05


class SearchOverflowError(Exception):
    """The search hit the per-filing match cap or found an oversized match."""


def regex_error(query: str) -> tuple[Optional[str], Optional[str]]:
    """``(error_code, message)`` if ``query`` won't compile in the engine search uses.

    Timed searches run on the third-party ``regex`` engine, whose syntax is a
    superset of ``re`` (``\\p{Lu}`` is valid there and not in ``re``), so
    validation has to use the same engine (QA finding P2-L5).
    """
    try:
        import regex as engine
    except ImportError:
        return "REGEX_UNAVAILABLE", "Timed regex search requires the 'regex' package from the 'ai' extra."
    try:
        engine.compile(query, engine.IGNORECASE)
    except engine.error as exc:
        return "INVALID_QUERY", f"Invalid regular expression: {exc}"
    return None, None


def query_digest(query: str) -> str:
    """SHA-256 of the query: what a search cursor binds to instead of the raw text (P2-L6)."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def search_cursor_query(query: str, regex: bool, include_all: bool) -> dict:
    """The cursor ``q`` identity for a search, without the optional raw query text."""
    return {
        "action": "search",
        "query_sha256": query_digest(query),
        "regex": bool(regex),
        "include_all": bool(include_all),
    }


def _location(attachment) -> str:
    from edgar._filings import Filing

    return Filing._attachment_location(attachment)


def _unsearched(attachment, reason: str) -> dict[str, Any]:
    return {
        "document": getattr(attachment, "document", None) or None,
        "sequence": str(getattr(attachment, "sequence_number", "") or ""),
        "reason": reason,
    }


def _document_matches(text: str, query: str, regex: bool, filename: str, location: str, remaining: int) -> list:
    matches = _grep_text(
        text,
        query,
        location,
        regex=regex,
        max_matches=remaining,
        max_match_chars=MAX_SEARCH_MATCH_CHARS,
        regex_timeout=REGEX_TIMEOUT_SECONDS if regex else None,
    )
    # A sentinel means the cap or the match-size limit was hit; the length
    # check is a defensive guard against an unbounded result.
    if any(match.overflowed for match in matches) or len(matches) > remaining:
        raise SearchOverflowError()
    return [
        {
            "location": match.location,
            "match": match.match,
            "context": match.context,
            "locator": {"document": filename, "char_offset": match.char_offset},
        }
        for match in matches
    ]


def _run_search(filing, attachments: list, query: str, regex: bool) -> tuple[list, list]:
    records: list[dict[str, Any]] = []
    unsearched: list[dict[str, Any]] = []
    for attachment in attachments:
        text, reason = read_text(filing, attachment)
        if reason is not None:
            unsearched.append(_unsearched(attachment, reason))
            continue
        filename = str(attachment.document)
        remaining = max(0, MAX_SEARCH_MATCHES - len(records))
        records.extend(_document_matches(text, query, regex, filename, _location(attachment), remaining))
    return records, unsearched


def search_documents(
    filing,
    attachments: list,
    query: str,
    regex: bool,
    *,
    document_filter: Optional[str],
    include_all: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """``(match records, unsearched documents, overflowed)`` for ``attachments``.

    Raises ``TimeoutError`` when a regex exceeds its per-document budget and
    ``ValidationError`` when a regex will not compile. Neither partial nor
    overflowed results are cached.
    """
    from edgar.ai.mcp.tools.continuation import results_cache

    cache_key = (
        "edgar_document:search",
        filing.accession_number,
        document_filter,
        query,
        bool(regex),
        bool(include_all),
        library_version(),
    )
    cached = results_cache.get(cache_key)
    if cached is not None:
        records, unsearched = cached
        return records, unsearched, False
    try:
        records, unsearched = _run_search(filing, attachments, query, regex)
    except SearchOverflowError:
        return [], [], True
    results_cache.put(cache_key, (records, unsearched))
    return records, unsearched, False


def search_fingerprint(records: list[dict[str, Any]], unsearched: list[dict[str, Any]]) -> str:
    parts = [json.dumps(record, sort_keys=True, default=str) for record in records]
    parts.extend(json.dumps(item, sort_keys=True, default=str) for item in unsearched)
    return fingerprint(parts)


def coverage_note(unsearched: list[dict[str, Any]]) -> Optional[str]:
    if not unsearched:
        return None
    count = len(unsearched)
    noun = "document" if count == 1 else "documents"
    return (
        f"{count} {noun} listed in unsearched_documents could not be read as text and were not searched. "
        "The match count, including zero, covers only the documents that were searched."
    )

