"""Filing attachment discovery, search and bounded document reading (``edgar_document``).

Identity (URLs, selectors, readable text) lives in ``document_identity``;
search lives in ``document_search``. This module owns the tool itself:
argument checks, cursor-only continuation (constraints rule 7b), paging and
response shapes.

A cursor alone is enough to continue a list, search or read: it names its
own filing (accession), document, query identity and filters. Omitted
arguments default to the cursor's values; supplied ones must agree with it,
else ``CURSOR_MISMATCH``. A search cursor binds a SHA-256 of the query rather
than the query itself, so a long query never breaks page 1 (P2-L6); the raw
query rides along only when it fits, and a cursor without it asks the caller
to pass the query again.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from edgar.ai.mcp.tools.base import ToolResponse, error, success, tool
from edgar.ai.mcp.tools.continuation import (
    TEXT_PAGE_CHARS,
    CursorError,
    canonical_accession,
    check_fingerprint,
    decode_cursor,
    fingerprint,
    paginate,
    paginate_text,
    peek_cursor,
    try_encode_cursor,
)
from edgar.ai.mcp.tools.document_identity import (
    eligible,
    filer_of,
    filing_source,
    identity,
    is_hidden_by_default,
    list_attachments,
    normalise_accession,
    parse_sec_url,
    read_text,
    resolve_attachment,
    resolve_url_filename,
    url_cik_matches,
)
from edgar.ai.mcp.tools.document_search import (
    coverage_note,
    regex_error,
    search_cursor_query,
    search_documents,
    search_fingerprint,
)
from edgar.exceptions import ValidationError

logger = logging.getLogger(__name__)

_LIST_TOOL = "edgar_document:list"
_SEARCH_TOOL = "edgar_document:search"
_READ_TOOL = "edgar_document:read"
_TOOL_BY_ACTION = {"list": _LIST_TOOL, "search": _SEARCH_TOOL, "read": _READ_TOOL}
_INCORPORATED_BY_REFERENCE_NOTE = (
    "Some exhibits referenced by this filing may be incorporated by reference from earlier SEC filings; "
    "they are not included in this filing's attachments. To find one, read the exhibit-index section "
    "or use edgar_text_search to locate the earlier filing."
)
_EVIDENCE_SCOPE_NOTE = (
    "If this is a BDC filing, it identifies evidence reported by the BDC and may not contain an underlying "
    "agreement for a portfolio borrower."
)


def _check_arguments(action, limit, document, around, cursor) -> Optional[ToolResponse]:
    if action not in _TOOL_BY_ACTION:
        return error("action must be one of: list, search, read.", error_code="INVALID_ACTION")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return error("limit must be a positive integer.", error_code="INVALID_LIMIT")
    if action == "list" and (document is not None or around is not None):
        return error(
            "action='list' lists every document; it does not accept document or around.",
            suggestions=["Omit document and around, or use action='read' or 'search' for one document."],
            error_code="INVALID_ARGUMENTS",
        )
    if action == "read" and around is not None and cursor:
        return error("around and cursor cannot be used together.", error_code="AROUND_CURSOR_CONFLICT")
    if action == "read" and around is not None and (
        not isinstance(around, dict)
        or not isinstance(around.get("document"), str)
        or isinstance(around.get("char_offset"), bool)
        or not isinstance(around.get("char_offset"), int)
        or around["char_offset"] < 0
    ):
        return error(
            "around must contain a document string and a non-negative integer char_offset.",
            error_code="INVALID_LOCATOR",
        )
    return None


def _peek_for_action(cursor: str, action: str) -> tuple[Optional[dict], Optional[ToolResponse]]:
    """The cursor's payload, checked only for shape and for this action's tool."""
    try:
        payload = peek_cursor(cursor)
    except CursorError as exc:
        return None, exc.to_response()
    if not isinstance(payload.get("acc"), str) or not payload["acc"]:
        return None, CursorError("Cursor does not name a filing.", error_code="INVALID_CURSOR").to_response()
    if payload.get("tool") != _TOOL_BY_ACTION[action]:
        return None, CursorError(
            "Cursor was issued for a different tool or action than this call.", error_code="CURSOR_MISMATCH"
        ).to_response()
    return payload, None


def _cursor_defaults(action, peeked, query, regex, include_all):
    """Fill omitted ``query``/``regex``/``include_all`` from the cursor (rule 7b)."""
    q = peeked.get("q") if peeked else None
    q = q if isinstance(q, dict) else {}
    include_all = bool(q.get("include_all", False)) if include_all is None else bool(include_all)
    regex = bool(q.get("regex", False)) if regex is None else bool(regex)
    if action == "search" and query is None and peeked:
        query = q.get("query")
        if not isinstance(query, str) or not query:
            return None, regex, include_all, error(
                "This cursor carries only a hash of its query because the query was too long to embed; "
                "pass the same query together with the cursor.",
                error_code="QUERY_REQUIRED",
            )
    return query, regex, include_all, None


def _check_query(action: str, query, regex: bool) -> Optional[ToolResponse]:
    if action != "search":
        return None
    if not isinstance(query, str) or not query:
        return error("query is required for action='search'.", error_code="QUERY_REQUIRED")
    if regex:
        code, message = regex_error(query)
        if code:
            return error(message, error_code=code)
    return None


def _source_input(accession_number, url, cursor_accession):
    """``(accession, sec_url, selected_by, error)`` from the caller's filing selectors."""
    sec_url = None
    if url is not None:
        sec_url, parse_error = parse_sec_url(url)
        if parse_error:
            return None, None, None, error(
                parse_error,
                suggestions=["Use an HTTPS URL on www.sec.gov under /Archives/edgar/data/<cik>/<accession>/."],
                error_code="INVALID_URL",
            )
    supplied = normalise_accession(accession_number) if accession_number else None
    if accession_number and supplied is None:
        return None, None, None, error(
            "The accession number must contain 18 digits, with or without SEC dashes.",
            error_code="INVALID_ACCESSION",
        )
    if supplied and sec_url and supplied != sec_url.accession:
        return None, None, None, error(
            "The accession_number and SEC URL identify different filings.", error_code="FILING_MISMATCH"
        )
    accession = sec_url.accession if sec_url else supplied
    if cursor_accession:
        if accession and accession != canonical_accession(cursor_accession):
            return None, None, None, CursorError(
                "Cursor was issued for a different filing than accession_number/url.", error_code="CURSOR_MISMATCH"
            ).to_response()
        accession = accession or canonical_accession(cursor_accession)
    if not accession:
        return None, None, None, error(
            "Provide accession_number, a valid SEC Archives URL, or a cursor from a previous page.",
            suggestions=["Use a dashed SEC accession number or an HTTPS www.sec.gov filing URL."],
            error_code="FILING_REQUIRED",
        )
    return accession, sec_url, "url" if url else "accession", None


def _load_filing(accession: str, sec_url):
    """``(filing, attachments, error)`` for an accession, checking a URL's CIK segment."""
    try:
        from edgar import find
        filing = find(search_id=accession)
    except Exception as exc:
        return None, None, error(f"Could not resolve SEC filing {accession}: {exc}", error_code="FILING_NOT_FOUND")
    if filing is None:
        return None, None, error(
            f"Filing {accession} was not found.",
            suggestions=["Check the accession number or use edgar_search to locate the filing."],
            error_code="FILING_NOT_FOUND",
        )
    try:
        attachments = list_attachments(filing)
    except RuntimeError as exc:
        return None, None, error(str(exc), error_code="ATTACHMENTS_UNAVAILABLE")
    if sec_url is not None and not url_cik_matches(filing, sec_url.cik):
        return None, None, error(
            f"The URL's CIK segment {sec_url.cik} is not a filer of accession {accession} "
            f"(filer CIK {int(filing.cik)}).",
            suggestions=["Use the filing's own URL, or pass accession_number without a URL."],
            error_code="FILING_MISMATCH",
        )
    return filing, attachments, None


def _same_attachment(first, second, message: str) -> Optional[ToolResponse]:
    if first.document != second.document:
        return error(message, error_code="DOCUMENT_MISMATCH")
    return None


def _explicit_selector(attachments, action, document, url_filename, around, filer):
    """The selector named by document/url/around, after checking they agree."""
    selector = document
    if url_filename and action in {"read", "search"}:
        url_attachment, failure = resolve_url_filename(attachments, url_filename, filer)
        if failure:
            return None, failure
        for other, label in ((document, "document selector"),
                             (around["document"] if action == "read" and around else None, "around.document")):
            if other is None:
                continue
            other_attachment, failure = resolve_attachment(attachments, other, filer)
            failure = failure or _same_attachment(
                other_attachment, url_attachment, f"The URL filename and {label} identify different attachments."
            )
            if failure:
                return None, failure
        selector = url_attachment.document
    if action == "read" and around is not None:
        if selector and selector != around["document"]:
            explicit, failure = resolve_attachment(attachments, selector, filer)
            if failure:
                return None, failure
            around_attachment, failure = resolve_attachment(attachments, around["document"], filer)
            failure = failure or _same_attachment(
                explicit, around_attachment, "document and around.document identify different attachments."
            )
            if failure:
                return None, failure
        selector = around["document"]
    return selector, None


def _select_attachment(attachments, action, document, url_filename, around, cursor_document, include_all, filer):
    selector, failure = _explicit_selector(attachments, action, document, url_filename, around, filer)
    if failure:
        return None, failure
    # A cursor was only ever issued for a document that passed the filter, so
    # a document taken from the cursor alone is not re-filtered (rule 7b).
    from_cursor = selector is None and cursor_document is not None
    selector = cursor_document if selector is None else selector
    if selector is None:
        return None, None
    attachment, failure = resolve_attachment(attachments, selector, filer)
    if failure:
        return None, failure
    if not include_all and not from_cursor and is_hidden_by_default(attachment):
        return None, error(
            f"Document {attachment.document!r} is hidden by the default XBRL and generated-file filter.",
            suggestions=["Retry with include_all=true to include XBRL, generated R and viewer files."],
            error_code="DOCUMENT_HIDDEN",
        )
    return attachment, None


def _page_records(*, action, accession, document, records, fp, cursor, limit, query):
    """One page of ``records`` plus the next cursor. ``query`` is the cursor ``q`` to bind."""
    tool_name = _TOOL_BY_ACTION[action]
    try:
        payload = decode_cursor(cursor, tool=tool_name, accession=accession, document=document, query=query) \
            if cursor else None
        if payload is not None:
            check_fingerprint(payload, fp)
    except CursorError as exc:
        return None, None, exc.to_response()

    offset = payload["off"] if payload is not None else 0
    page_items, meta = paginate(records, offset=offset, limit=limit)
    next_cursor = None
    if meta["remaining"]:
        next_cursor, cursor_error = _encode_next(tool_name, accession, offset + meta["returned"], fp, document, query)
        if cursor_error:
            return None, None, cursor_error
    return page_items, {**meta, "next_cursor": next_cursor}, None


def _encode_next(tool_name, accession, offset, fp, document, query):
    """Encode a cursor, embedding a search's raw query text only when it fits."""
    raw_query = query.get("query") if isinstance(query, dict) else None
    if raw_query is not None:
        cursor, failure = try_encode_cursor(
            tool=tool_name, accession=accession, offset=offset, fp=fp, document=document, query=query
        )
        if failure is None:
            return cursor, None
        query = {key: value for key, value in query.items() if key != "query"}
    return try_encode_cursor(tool=tool_name, accession=accession, offset=offset, fp=fp, document=document, query=query)


def _search_cursor_identity(query: str, regex: bool, include_all: bool, cursor: Optional[str]):
    """The cursor ``q`` to decode against: the cursor's own, once its hash identity matches."""
    base = search_cursor_query(query, regex, include_all)
    if not cursor:
        return {**base, "query": query}, None
    q = peek_cursor(cursor).get("q")
    cursor_base = {key: value for key, value in q.items() if key != "query"} if isinstance(q, dict) else q
    if cursor_base != base:
        return None, CursorError(
            "Cursor was issued for a different query than this call.", error_code="CURSOR_MISMATCH"
        ).to_response()
    return q, None


def _json_record(record: dict) -> str:
    return json.dumps(record, sort_keys=True, default=str)


def _common(filer: dict, source: dict) -> dict:
    return {
        "incorporated_by_reference_note": _INCORPORATED_BY_REFERENCE_NOTE,
        "evidence_scope_note": _EVIDENCE_SCOPE_NOTE,
        "filer": filer,
        "source": source,
    }


def _list_response(accession, attachments, include_all, cursor, limit, common) -> ToolResponse:
    records = [identity(a) for a in eligible(attachments, include_all)]
    fp = fingerprint(_json_record(record) for record in records)
    items, page, failure = _page_records(
        action="list", accession=accession, document=None, records=records, fp=fp,
        cursor=cursor, limit=limit, query={"action": "list", "include_all": include_all},
    )
    if failure:
        return failure
    return success({"accession_number": accession, "documents": items, "page": page, **common})


def _search_error(exc: Exception) -> ToolResponse:
    if isinstance(exc, TimeoutError):
        return error(
            "Regex search timed out at the 50 ms per-document limit; partial results were discarded.",
            suggestions=[
                "Simplify the regular expression to avoid excessive backtracking.",
                "Narrow the search to one exact document or use a literal text query.",
            ],
            error_code="REGEX_TIMEOUT",
        )
    return error(str(exc), error_code="INVALID_QUERY")


def _overflow_error() -> ToolResponse:
    return error(
        "Search exceeded the limit of 1,000 matches per filing or found a match longer than 2,048 characters.",
        suggestions=[
            "Narrow the query or regular expression to reduce the number or size of matches.",
            "Search one exact document at a time by setting document to its filename or sequence.",
        ],
        error_code="QUERY_TOO_BROAD",
    )


def _search_response(filing, accession, attachments, selected, request, common) -> ToolResponse:
    query, regex, include_all, cursor, limit = request
    if selected is not None:
        _, reason = read_text(filing, selected)
        if reason is not None:
            # P2-H1: an unreadable document is never reported as "0 matches".
            return success({
                "accession_number": accession,
                "document": identity(selected),
                "unreadable_reason": reason,
                "matches": None,
                "page": None,
                "unsearched_documents": [{
                    "document": selected.document or None,
                    "sequence": str(getattr(selected, "sequence_number", "") or ""),
                    "reason": reason,
                }],
                **common,
            })
    q, failure = _search_cursor_identity(query, regex, include_all, cursor)
    if failure:
        return failure
    targets = [selected] if selected is not None else eligible(attachments, include_all)
    document = selected.document if selected is not None else None
    try:
        records, unsearched, overflowed = search_documents(
            filing, targets, query, regex, document_filter=document, include_all=include_all
        )
    except (TimeoutError, ValidationError) as exc:
        return _search_error(exc)
    if overflowed:
        return _overflow_error()
    items, page, failure = _page_records(
        action="search", accession=accession, document=document, records=records,
        fp=search_fingerprint(records, unsearched), cursor=cursor, limit=limit, query=q,
    )
    if failure:
        return failure
    data = {"accession_number": accession, "matches": items, "page": page, "unsearched_documents": unsearched}
    note = coverage_note(unsearched)
    if note:
        data["coverage_note"] = note
    return success({**data, **common})


def _read_offset(text: str, payload: Optional[dict], around: Optional[dict]):
    if payload is not None:
        return payload["off"], None
    if around is None:
        return 0, None
    char_offset = around["char_offset"]
    if char_offset > len(text):
        return None, error(
            f"around.char_offset {char_offset} is beyond the document's {len(text)} characters.",
            error_code="INVALID_LOCATOR",
        )
    return max(0, min(char_offset - TEXT_PAGE_CHARS // 2, len(text) - TEXT_PAGE_CHARS)), None


def _read_response(filing, accession, selected, around, cursor, common) -> ToolResponse:
    if selected is None:
        return error(
            "A document selector is required for action='read'.",
            suggestions=["Provide document as a sequence number, filename or exhibit type, or use around."],
            error_code="DOCUMENT_REQUIRED",
        )
    text, reason = read_text(filing, selected)
    if reason:
        return success({
            "accession_number": accession, "document": identity(selected), "unreadable_reason": reason, **common,
        })

    fp = fingerprint([text])
    try:
        payload = decode_cursor(cursor, tool=_READ_TOOL, accession=accession, document=selected.document) \
            if cursor else None
        if payload is not None:
            check_fingerprint(payload, fp)
    except CursorError as exc:
        return exc.to_response()
    offset, failure = _read_offset(text, payload, around)
    if failure:
        return failure

    page_text, meta = paginate_text(text, offset=offset, budget=TEXT_PAGE_CHARS)
    next_cursor = None
    if meta["next_offset"] is not None:
        next_cursor, failure = try_encode_cursor(
            tool=_READ_TOOL, accession=accession, document=selected.document, offset=meta["next_offset"], fp=fp,
        )
        if failure:
            return failure
    return success({
        "accession_number": accession,
        "document": identity(selected),
        "text": page_text,
        "page": {
            "offset": offset,
            "total_chars": meta["total_chars"],
            "remaining_chars": meta["remaining_chars"],
            "next_cursor": next_cursor,
        },
        **common,
    })


_EXAMPLE = "<!-- MCP_TOOL_CALL_EXAMPLE -->\n```json\n{}\n```"


@tool(
    name="edgar_document",
    description=(
        "List, search and read exact documents filed with an SEC filing. Use accession_number or an HTTPS "
        "www.sec.gov Archives URL, which binds the accession and filename and is validated as identity rather "
        "than fetched directly; the URL's CIK must be a filer of the accession, and an index-page or "
        ".txt URL names the filing rather than one document. Responses report how the filing was chosen as "
        'source.selected_by: "accession", or selected_by: "url" when a URL chose it. '
        "Select attachments by sequence, exact filename, primary (sequence 1) or exhibit type; ambiguous "
        "types return candidate identities. list hides XBRL, generated R pages and XBRL viewer files unless "
        "include_all is true. Search locators can be passed to read via around. Reads return "
        "at most 6,000 characters per page. Searches return "
        "at most 1,000 matches per filing and reject matches longer than 2,048 characters. Regex searches "
        "have a 50 ms per-document time limit. Search never reports zero matches for text it could not read: "
        "an unreadable selected document returns unreadable_reason, and an all-documents search lists skipped "
        "attachments in unsearched_documents. A cursor alone continues a list, search or read; omitted "
        "arguments default to the cursor's values and supplied ones must match it (else CURSOR_MISMATCH). "
        "A BDC filing "
        "does not necessarily contain a portfolio borrower's agreement. "
        "Search results carry {document, char_offset} locators for bounded reads. "
        "Exhibits incorporated by reference may be in an earlier filing: read the exhibit-index section or "
        "use edgar_text_search.\n\n"
        + "\n\n".join(_EXAMPLE.format(example) for example in (
            '{"tool":"edgar_document","arguments":{"action":"list","accession_number":"0001628280-26-050307"}}',
            '{"tool":"edgar_document","arguments":{"action":"search","accession_number":"0001628280-26-050307","document":"[exact-filename-from-list]","query":"loan agreement"}}',
            '{"tool":"edgar_document","arguments":{"action":"read","accession_number":"0001628280-26-050307","document":"[exact-filename-from-list]"}}',
            '{"tool":"edgar_document","arguments":{"action":"read","accession_number":"0001628280-26-050307","document":"[exact-filename-from-list]","cursor":"<page.next_cursor>"}}',
            '{"tool":"edgar_document","arguments":{"action":"read","url":"https://www.sec.gov/Archives/edgar/data/320193/000032019325000073/a10-qexhibit32106282025.htm"}}',
        ))
    ),
    params={
        "action": {"type": "string", "enum": ["list", "search", "read"], "description": "Document action."},
        "accession_number": {"type": "string", "description": "SEC accession number, dashed or undashed."},
        "url": {"type": "string", "description": "HTTPS www.sec.gov filing or document URL under /Archives/."},
        "document": {
            "type": "string",
            "description": "Exact sequence, filename, 'primary' or exhibit type prefix (read/search only). "
                           "With a cursor, defaults to the cursor's document.",
        },
        "query": {
            "type": "string",
            "description": "Text or regular expression to search within filing documents. With a cursor, "
                           "defaults to the cursor's query when the cursor carries it.",
        },
        "regex": {
            "type": "boolean",
            "description": "Treat query as a regular expression with a 50 ms per-document time limit. "
                           "With a cursor, defaults to the cursor's value.",
            "default": False,
        },
        "include_all": {
            "type": "boolean",
            "description": "Include XBRL, generated R files and XBRL viewer files. With a cursor, defaults to "
                           "the cursor's value.",
            "default": False,
        },
        "around": {
            "type": "object",
            "description": "Start a read centered on a search locator {document, char_offset}.",
            "properties": {
                "document": {"type": "string"},
                "char_offset": {"type": "integer", "minimum": 0},
            },
            "required": ["document", "char_offset"],
        },
        "cursor": {
            "type": "string",
            "description": "Continuation cursor from a prior list, search or read page. The cursor alone is "
                           "enough to fetch the next page.",
        },
        "limit": {
            "type": "integer",
            "description": "Records per list/search page (default 20, max 50).",
            "default": 20,
        },
    },
    required=["action"],
)
async def edgar_document(
    action: str,
    accession_number: Optional[str] = None,
    url: Optional[str] = None,
    document: Optional[str] = None,
    query: Optional[str] = None,
    regex: Optional[bool] = None,
    include_all: Optional[bool] = None,
    around: Optional[dict] = None,
    cursor: Optional[str] = None,
    limit: int = 20,
) -> ToolResponse:
    """List, search or read attachments while retaining exact SEC document identity."""
    failure = _check_arguments(action, limit, document, around, cursor)
    if failure:
        return failure
    limit = min(limit, 50)

    peeked = None
    if cursor:
        peeked, failure = _peek_for_action(cursor, action)
        if failure:
            return failure
    query, regex, include_all, failure = _cursor_defaults(action, peeked, query, regex, include_all)
    failure = failure or _check_query(action, query, regex)
    if failure:
        return failure

    accession, sec_url, selected_by, failure = _source_input(
        accession_number, url, peeked["acc"] if peeked else None
    )
    if failure:
        return failure
    filing, attachments, failure = _load_filing(accession, sec_url)
    if failure:
        return failure

    filer = filer_of(filing)
    if action == "list":
        return _list_response(
            accession, attachments, include_all, cursor, limit, _common(filer, filing_source(filing, selected_by))
        )

    selected, failure = _select_attachment(
        attachments, action, document, sec_url.filename if sec_url else None, around,
        peeked.get("doc") if peeked else None, include_all, filer,
    )
    if failure:
        return failure
    common = _common(filer, filing_source(filing, selected_by, selected))
    if action == "search":
        return _search_response(
            filing, accession, attachments, selected, (query, regex, include_all, cursor, limit), common
        )
    return _read_response(filing, accession, selected, around, cursor, common)
