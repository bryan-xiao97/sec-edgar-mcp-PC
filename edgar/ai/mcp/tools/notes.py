"""
Notes Tool (edgar_notes)

Drill into the notes and disclosures behind financial statement numbers.
Given a company and topic, returns structured note content: narrative text,
table data, which statement line items the note expands, and child tables.

This is the tool to call when an AI needs to explain WHY a number is what
it is — not just what the number is.

Filing selection goes through the shared resolver
(``edgar.ai.mcp.tools.selection.resolve_report_filing``): ``accession_number``
takes precedence, then ``period`` (must exactly match an original filing's
``period_of_report``), then the latest original filing. Every response
carries a ``source`` provenance block (``format_source``).

A matched note's table rows and narrative ``context`` text are both capped
per page (record rows: ``limit``, default 20; text: ``TEXT_PAGE_CHARS``).
When more remains, the table/context carries a ``next_cursor``; passing that
cursor back returns only that one continued item. A cursor ALONE is enough
(constraints rule 7b): the filing it continues comes from the cursor's own
accession, and an omitted ``topic``/``detail`` defaults to the value the
cursor was minted under. A ``topic``/``detail`` supplied together with the
cursor must equal that value, or it is ``CURSOR_MISMATCH`` — a client that
changes topic between calls never silently gets the other topic's page.
A note's table pages under cursor tool ``"edgar_notes:table"``; its context
text under ``"edgar_notes:context"``. A response can emit cursors of both
kinds at once (multiple matched notes, multiple tables), so continuing has
to read the *kind* out of the cursor before it can validate the rest. See
``_continue_notes``.

Continuation never keeps the filing's parsed report object (and therefore
never keeps its ``_xbrl``) alive across calls (P2-M4): only each matched
table's serialized rows and each note's context text are cached, in
``continuation.text_cache`` (which has a byte budget), keyed by
``(accession, note number, table index)`` / ``(accession, note number,
detail)``. A cache miss during continuation re-parses the filing once to
rebuild just that one table/context, exactly as a first call would.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from edgar.ai.mcp.tools.base import (
    ToolResponse,
    _cell_number,
    error,
    format_source,
    success,
    tool,
)
from edgar.ai.mcp.tools.continuation import (
    CursorError,
    check_fingerprint,
    fingerprint,
    paginate,
    paginate_text,
    peek_cursor,
    resolve_cursor_accession,
    try_decode_cursor,
    try_encode_cursor,
)

logger = logging.getLogger(__name__)

_TABLE_TOOL = "edgar_notes:table"
_CONTEXT_TOOL = "edgar_notes:context"


def _table_cache_key(accession: str, note_number: int, table_index: int):
    return ("notes:table", accession, note_number, table_index)


def _context_cache_key(accession: str, note_number: int, detail: str):
    return ("notes:context", accession, note_number, detail)


def _records_size(records: list[dict]) -> int:
    return len(json.dumps(records, default=str).encode("utf-8"))


# =============================================================================
# Row / table serialization
# =============================================================================

def _serialize_records(df) -> list[dict]:
    """DataFrame rows to JSON-safe dicts: missing -> None, Decimal/numpy -> native."""
    return [{k: _cell_number(v) for k, v in row.items()} for _, row in df.iterrows()]


def _table_fingerprint(records: list[dict]) -> str:
    """Fingerprint over a table's rows, serialized as JSON (per the task brief)."""
    return fingerprint(json.dumps(r, sort_keys=True, default=str) for r in records)


def _build_table_info(note, table_index: int, table, *, topic: Optional[str], limit: int, accession: str) -> dict:
    """One table entry: role/title, plus (for detail='full') a first page of rows."""
    table_info: dict[str, Any] = {"role": table.role_or_type}
    try:
        rendered = table.render()
        if rendered:
            table_info["title"] = rendered.title
    except Exception:
        pass

    try:
        df = table.to_dataframe()
    except Exception:
        df = None
    if df is None or df.empty:
        return table_info

    records = _serialize_records(df)
    rows_total = len(records)
    table_info["rows"] = rows_total  # existing key's meaning: total rows (unchanged)
    table_info["rows_total"] = rows_total  # new key, same value
    table_info["columns"] = list(df.columns[:10])

    page_items, page_meta = paginate(records, offset=0, limit=limit)
    table_info["data"] = page_items

    if page_meta["remaining"] > 0:
        fp = _table_fingerprint(records)
        query = {"topic": topic, "note": note.number, "table": table_index}
        next_cursor, err = try_encode_cursor(
            tool=_TABLE_TOOL, accession=accession, offset=page_meta["returned"], fp=fp, query=query
        )
        if err is None:
            table_info["next_cursor"] = next_cursor
            # Cache the serialized rows (never the note/report object itself,
            # which would pin the parsed XBRL -- P2-M4) so a later
            # continuation call can page through them without re-parsing.
            from edgar.ai.mcp.tools.continuation import text_cache

            text_cache.put(
                _table_cache_key(accession, note.number, table_index),
                {"title": note.title, "records": records},
                size_bytes=_records_size(records),
            )
        else:
            logger.warning("Could not build table cursor for note %s: %s", note.number, err)

    return table_info


def _build_context_page(note, text: str, *, topic: Optional[str], detail: str, accession: str) -> tuple[str, dict]:
    """First page of a note's context text, plus its context_page metadata block."""
    page_text, meta = paginate_text(text, offset=0)
    context_page: dict[str, Any] = {
        "offset": 0,
        "total_chars": meta["total_chars"],
        "remaining_chars": meta["remaining_chars"],
        "next_cursor": None,
    }
    if meta["next_offset"] is not None:
        fp = fingerprint([text])
        query = {"topic": topic, "note": note.number, "detail": detail}
        next_cursor, err = try_encode_cursor(
            tool=_CONTEXT_TOOL, accession=accession, offset=meta["next_offset"], fp=fp, query=query
        )
        if err is None:
            context_page["next_cursor"] = next_cursor
            # Cache the context text itself (never the note/report object --
            # P2-M4) so a later continuation call can page through it
            # without re-parsing.
            from edgar.ai.mcp.tools.continuation import text_cache

            text_cache.put(
                _context_cache_key(accession, note.number, detail),
                {"title": note.title, "text": text},
                size_bytes=len(text.encode("utf-8")),
            )
        else:
            logger.warning("Could not build context cursor for note %s: %s", note.number, err)
    return page_text, context_page


def _build_note_data(note, *, detail: str, topic: Optional[str], limit: int, accession: str) -> dict:
    """Build a structured dict from a Note object."""
    data: dict[str, Any] = {
        "number": note.number,
        "title": note.title,
    }

    if note.expands:
        data["expands"] = note.expands
    if note.expands_statements:
        data["expands_statements"] = note.expands_statements

    if note.tables:
        tables = []
        for idx, t in enumerate(note.tables):
            if detail == "full":
                tables.append(_build_table_info(note, idx, t, topic=topic, limit=limit, accession=accession))
            else:
                table_info: dict[str, Any] = {"role": t.role_or_type}
                try:
                    rendered = t.render()
                    if rendered:
                        table_info["title"] = rendered.title
                except Exception:
                    pass
                tables.append(table_info)
        data["tables"] = tables

    context_text = note.to_context(detail=detail)
    page_text, context_page = _build_context_page(note, context_text, topic=topic, detail=detail, accession=accession)
    data["context"] = page_text
    data["context_page"] = context_page

    return data


# =============================================================================
# Continuation
# =============================================================================

def _load_note(filing, note_number: int):
    """A fresh note object for a continuation cache miss.

    Parses `filing.obj()` -- there is no cross-call cache of the report
    object itself (P2-M4: caching it would keep its `_xbrl` alive) -- and
    returns `notes[note_number]`, or `None` if the filing has no such note
    (or fails to parse at all).
    """
    try:
        obj = filing.obj()
    except Exception:
        return None
    notes = getattr(obj, "notes", None)
    if not notes:
        return None
    return notes[note_number] if isinstance(note_number, int) else None


def _continue_table(filing, accession: str, note_number: int, table_index, query: dict, payload: dict, limit: int):
    """Continue a table cursor. Returns `("table", page_items, page_block, note_title)`
    or a ToolResponse error.

    Checks `text_cache` for this note/table's already-serialized rows
    BEFORE loading the filing (P2-M4) -- a hit needs no note/report object
    at all. A miss re-parses once via `_load_note` and caches the result
    for the next page.
    """
    if not isinstance(table_index, int) or table_index < 0:
        return CursorError(
            "Cursor refers to a table that is not in this note.", error_code="CURSOR_MISMATCH"
        ).to_response()

    from edgar.ai.mcp.tools.continuation import text_cache

    cache_key = _table_cache_key(accession, note_number, table_index)
    cached = text_cache.get(cache_key)
    if cached is None:
        note = _load_note(filing, note_number)
        if note is None or table_index >= len(note.tables):
            return CursorError(
                "Cursor refers to a table that is not in this note.", error_code="CURSOR_MISMATCH"
            ).to_response()
        try:
            df = note.tables[table_index].to_dataframe()
        except Exception:
            df = None
        if df is None or df.empty:
            return CursorError(
                "This note's table no longer has rows to page through.", error_code="CURSOR_STALE"
            ).to_response()
        records = _serialize_records(df)
        cached = {"title": note.title, "records": records}
        text_cache.put(cache_key, cached, size_bytes=_records_size(records))

    records = cached["records"]
    note_title = cached["title"]
    fp = _table_fingerprint(records)
    try:
        check_fingerprint(payload, fp)
    except CursorError as exc:
        return exc.to_response()

    offset = payload["off"]
    page_items, page_meta = paginate(records, offset=offset, limit=limit)

    next_cursor = None
    if page_meta["remaining"] > 0:
        next_cursor, err = try_encode_cursor(
            tool=_TABLE_TOOL, accession=accession, offset=offset + page_meta["returned"], fp=fp, query=query
        )
        if err is not None:
            return err

    page_block = {**page_meta, "next_cursor": next_cursor}
    return "table", page_items, page_block, note_title


def _continue_context(filing, accession: str, note_number: int, query: dict, payload: dict):
    """Continue a context cursor. Returns `("context", page_text, page_block, note_title)`
    or a ToolResponse error.

    Same cache-before-parse shape as `_continue_table` (P2-M4).
    """
    detail = query.get("detail") or "standard"

    from edgar.ai.mcp.tools.continuation import text_cache

    cache_key = _context_cache_key(accession, note_number, detail)
    cached = text_cache.get(cache_key)
    if cached is None:
        note = _load_note(filing, note_number)
        if note is None:
            return CursorError(
                "Cursor refers to a note that is not in this filing.", error_code="CURSOR_MISMATCH"
            ).to_response()
        text = note.to_context(detail=detail)
        cached = {"title": note.title, "text": text}
        text_cache.put(cache_key, cached, size_bytes=len(text.encode("utf-8")))

    text = cached["text"]
    note_title = cached["title"]
    fp = fingerprint([text])
    try:
        check_fingerprint(payload, fp)
    except CursorError as exc:
        return exc.to_response()

    offset = payload["off"]
    page_text, meta = paginate_text(text, offset=offset)

    next_cursor = None
    if meta["next_offset"] is not None:
        next_cursor, err = try_encode_cursor(
            tool=_CONTEXT_TOOL, accession=accession, offset=meta["next_offset"], fp=fp, query=query
        )
        if err is not None:
            return err

    page_block = {
        "offset": offset,
        "total_chars": meta["total_chars"],
        "remaining_chars": meta["remaining_chars"],
        "next_cursor": next_cursor,
    }
    return "context", page_text, page_block, note_title


def _continue_notes(
    cursor: str,
    filing,
    selected_by: str,
    accession: str,
    limit: int,
    topic: Optional[str],
    detail: Optional[str],
) -> Any:
    """Resolve `cursor` to exactly the note/table or note/context it continues.

    Cursor-only continuation (constraints rule 7b): `topic` and (for a
    context cursor) `detail` default to the value the cursor itself was
    minted under (`raw_query`) when the current call omits them (`None`).
    A caller that supplies a differing value still fails
    `decode_cursor`'s query comparison with `CURSOR_MISMATCH` -- the
    "expected" query is built from whichever of (current call, cursor) each
    field actually came from, not read wholesale from the cursor and
    checked against itself.

    The cursor's `tool` is peeked out first (see `peek_cursor`'s docstring
    for why): this response can emit a table cursor or a context cursor, so
    which one a given cursor is has to be read out of it before it can be
    validated. The cursor's own `q.note` and (for a table cursor) `q.table`
    are also read from the peek -- those are positional, identifying WHICH
    note/table a cursor points at, and the call has no other way to supply
    them.
    """
    try:
        raw = peek_cursor(cursor)
    except CursorError as exc:
        return exc.to_response()

    tool_name = raw.get("tool")
    raw_query = raw.get("q") if isinstance(raw.get("q"), dict) else {}
    note_number = raw_query.get("note")

    if not isinstance(note_number, int) or tool_name not in (_TABLE_TOOL, _CONTEXT_TOOL):
        return CursorError(
            "Cursor refers to a note that is not in this filing.", error_code="CURSOR_MISMATCH"
        ).to_response()

    effective_topic = topic if topic is not None else raw_query.get("topic")

    if tool_name == _TABLE_TOOL:
        table_index = raw_query.get("table")
        expected_query = {"topic": effective_topic, "note": note_number, "table": table_index}
    else:
        effective_detail = detail if detail is not None else (raw_query.get("detail") or "standard")
        expected_query = {"topic": effective_topic, "note": note_number, "detail": effective_detail}

    payload, err = try_decode_cursor(cursor, tool=tool_name, accession=accession, query=expected_query)
    if err is not None:
        return err

    if tool_name == _TABLE_TOOL:
        page_result = _continue_table(filing, accession, note_number, table_index, expected_query, payload, limit)
    else:
        page_result = _continue_context(filing, accession, note_number, expected_query, payload)

    if isinstance(page_result, ToolResponse):
        return page_result

    key, page_value, page_block, note_title = page_result
    result = {
        "source": format_source(filing, selected_by),
        "note": {"number": note_number, "title": note_title},
        key: page_value,
        "page": page_block,
    }
    return success(result, next_steps=["Omit the cursor to see other matched notes"])


# =============================================================================
# Tool
# =============================================================================

@tool(
    name="edgar_notes",
    description="""Drill into the notes and disclosures behind financial statement numbers. Use this when you need to explain WHY a number is what it is — debt terms, revenue recognition policies, lease schedules, contingencies, etc.

Returns the note's narrative text, which statement line items it explains, and structured table data.

Examples:
- What does Apple's debt note say? topic="debt", identifier="AAPL"
- Revenue recognition policy: topic="revenue", identifier="MSFT"
- All notes overview: identifier="TSLA" (no topic = table of contents)
- Notes from a chosen period: topic="debt", identifier="ARCC", form="10-Q", period="2026-06-30"
- Notes from a chosen filing by accession: topic="debt", accession_number="0001628280-26-050307"
- Continue a table or note context: topic="debt", identifier="ARCC", form="10-Q", period="2026-06-30", detail="standard", cursor="<table-or-context.next_cursor>"
- Continue with the cursor alone (identifier/topic/detail are optional and default to the cursor's own values): cursor="<table-or-context.next_cursor>"

<!-- MCP_TOOL_CALL_EXAMPLE -->
```json
{"tool":"edgar_notes","arguments":{"topic":"debt","identifier":"ARCC","form":"10-Q","period":"2026-06-30","detail":"standard","limit":20}}
```

<!-- MCP_TOOL_CALL_EXAMPLE -->
```json
{"tool":"edgar_notes","arguments":{"topic":"debt","identifier":"ARCC","form":"10-Q","period":"2026-06-30","detail":"standard","limit":20,"cursor":"<table-or-context.next_cursor>"}}
```""",
    params={
        "identifier": {
            "type": "string",
            "description": "Company ticker (AAPL), CIK (320193), or name. Optional when accession_number is given."
        },
        "topic": {
            "type": "string",
            "description": "Note topic to search for (e.g., 'debt', 'revenue', 'leases', 'contingencies'). Omit for table of contents."
        },
        "form": {
            "type": "string",
            "description": "Filing form type (default: 10-K). Use 10-Q for quarterly notes.",
            "default": "10-K"
        },
        "detail": {
            "type": "string",
            "enum": ["minimal", "standard", "full"],
            "description": "Detail level: minimal (titles only), standard (context + tables), full (includes DataFrame data)",
            "default": "standard"
        },
        "accession_number": {
            "type": "string",
            "description": "Exact SEC accession number (NNNNNNNNNN-NN-NNNNNN) pinning a specific filing. Takes precedence over period."
        },
        "period": {
            "type": "string",
            "description": "Reporting period (YYYY-MM-DD) that must exactly match the chosen filing's period_of_report."
        },
        "cursor": {
            "type": "string",
            "description": "Continuation cursor from a previous response's table/context next_cursor, to fetch the next page of that one item. The cursor ALONE is enough -- it selects its own filing, topic, and (for a note's context) detail level. identifier/accession_number/topic/detail, if also given, must agree with the cursor (identifier's CIK; accession_number exactly; topic/detail exactly), else CURSOR_MISMATCH or SELECTION_MISMATCH. form/period are ignored when a cursor is given."
        },
        "limit": {
            "type": "integer",
            "description": "Max table rows per page (default 20, max 50).",
            "default": 20
        }
    },
    required=[]
)
async def edgar_notes(
    identifier: Optional[str] = None,
    topic: Optional[str] = None,
    form: str = "10-K",
    detail: Optional[str] = None,
    accession_number: Optional[str] = None,
    period: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = 20,
) -> Any:
    """Get notes and disclosures for a company filing."""
    limit = max(1, min(50, limit))

    if not identifier and not accession_number and not cursor:
        return error(
            "identifier or accession_number is required",
            suggestions=[
                "Provide a company ticker (AAPL), CIK, or name as identifier",
                "Or provide accession_number to pin an exact filing",
                "Or provide the cursor from a previous page to continue",
            ],
            error_code="INVALID_ARGUMENTS",
        )

    # Cursor-only continuation (constraints rule 7b): a cursor alone names
    # its own filing by accession, so accession_number/period do not have
    # to be re-supplied just to continue. A differing accession_number is
    # CURSOR_MISMATCH, checked before any filing selection.
    select_accession, err = resolve_cursor_accession(cursor, accession_number)
    if err is not None:
        return err
    select_period = None if cursor else period

    # Imported locally (not at module import time) so tests can monkeypatch
    # edgar.ai.mcp.tools.selection.resolve_report_filing and have it take
    # effect here -- same convention fund.py uses for filing selection.
    from edgar.ai.mcp.tools.selection import FilingSelectionError, resolve_report_filing

    try:
        resolved = resolve_report_filing(
            identifier=identifier, form=form, accession_number=select_accession, period=select_period
        )
    except FilingSelectionError as exc:
        return exc.to_response()

    filing = resolved.filing
    selected_by = resolved.selected_by
    accession = filing.accession_number

    if cursor:
        return _continue_notes(cursor, filing, selected_by, accession, limit, topic, detail)

    detail = detail or "standard"

    try:
        obj = filing.obj()
    except Exception as e:
        return error(
            f"Could not parse {filing.form} filing: {e}",
            suggestions=["The filing may not have XBRL data", "Try a different filing"],
            error_code="INTERNAL_ERROR",
        )

    if not hasattr(obj, "notes"):
        return error(
            f"{type(obj).__name__} does not support notes",
            suggestions=["Notes are available for 10-K and 10-Q filings"],
            error_code="INVALID_ARGUMENTS",
        )

    notes = obj.notes
    report_period = getattr(obj, "period_of_report", None)
    if not notes or len(notes) == 0:
        return error(
            f"No notes found in {filing.form} filing for {filing.company}",
            suggestions=["The filing may not have FilingSummary.xml", "Try a different filing period"],
            error_code="FILING_NOT_FOUND",
        )

    result: dict[str, Any] = {
        "company": filing.company,
        "form": filing.form,
        "filed": str(filing.filing_date),
        "period": report_period,
        "total_notes": len(notes),
        "source": format_source(filing, selected_by),
    }

    if topic:
        matched = notes.search(topic)
        if not matched:
            available = [n.short_name for n in notes]
            return error(
                f"No notes matching '{topic}' in {filing.form} filing",
                suggestions=[f"Available notes: {', '.join(available[:10])}"],
                error_code="FILING_NOT_FOUND",
            )

        result["topic"] = topic
        result["matched_notes"] = len(matched)
        result["notes"] = [
            _build_note_data(note, detail=detail, topic=topic, limit=limit, accession=accession)
            for note in matched
        ]

        next_steps = []
        primary = matched[0]
        if primary.expands_statements:
            stmts = primary.expands_statements
            next_steps.append(f"Use edgar_company to see the {', '.join(stmts)} for {filing.company}")
        if len(matched) > 1:
            other_titles = [n.title for n in matched[1:3]]
            next_steps.append(f"Related notes also matched: {', '.join(other_titles)}")

        return success(result, next_steps=next_steps)

    # Table of contents — list all notes
    result["notes"] = []
    for note in notes:
        entry = {
            "number": note.number,
            "title": note.title,
            "tables": note.table_count,
            "policies": len(note.policies),
            "details": len(note.details),
        }
        if detail != "minimal" and note.expands:
            entry["expands"] = note.expands[:5]
        result["notes"].append(entry)

    return success(
        result,
        next_steps=[
            "Use edgar_notes with topic='debt' (or any note title) to drill into a specific note",
            f"Use edgar_company with include=['financials'] for {filing.company} financial data",
        ],
    )
