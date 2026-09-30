"""Bounded, honest attachment search for ``edgar_document``.

Search reads every document through ``document_identity.read_text`` -- the
same cached text ``read`` pages -- so a locator's ``char_offset`` always
indexes the text a follow-up read returns, and a document whose text cannot
be read is reported in ``unsearched_documents`` rather than silently counted
as "no matches" (QA finding P2-H1).

That text is rendered markdown, which backslash-escapes punctuation
(``Sarbanes\\-Oxley``, ``15\\(d\\)``, ``U\\.S\\.``) and wraps lines. Matching it
directly misses body text a reader plainly sees (Q4 review findings I1 and
Gap A), so queries run against a *reader view* of the rendered text:

- Escapes are removed exactly where ``edgar.documents``' renderer adds them:
  the characters ``\\ ` * _ { } [ ] ( ) # + - . !`` in prose. Table text is
  never escaped by the renderer, so a backslash there is real (Gap B).
  Table rows are recognized by the pipe renderer's row grammar, not by a
  leading ``|`` alone (round 3): see ``_table_rows``.
- For literal queries, every run of whitespace (spaces, tabs, newlines,
  non-breaking spaces) becomes one space, and so does every run in the
  query. ``|`` is not whitespace, so a literal match never spans a table-cell
  boundary unless the query itself contains ``|``.
- Regex queries keep their own semantics: they see the unescaped text with
  its whitespace intact (``\\s+`` is how a regex tolerates wrapping).

A view character stands for one rendered span (a character, a backslash
plus the character it escapes, or a whole whitespace run). Spans are
contiguous, so a view offset ``x`` maps back to the rendered offset
``x + (extra rendered characters of the spans before x)``, which is exact at
both ends of a match. Locators, ``match`` and ``context`` are therefore
verbatim slices of the text ``read`` pages; ``match_text`` is the match as
the view shows it.
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from typing import Any, NamedTuple, Optional

from edgar.ai.mcp.tools.continuation import fingerprint
from edgar.ai.mcp.tools.document_identity import library_version, read_text
from edgar.search.grep import _grep_text

MAX_SEARCH_MATCHES = 1_000
MAX_SEARCH_MATCH_CHARS = 2_048
REGEX_TIMEOUT_SECONDS = 0.05
CONTEXT_CHARS = 100
# Exactly what edgar.documents' markdown renderer escapes (``_escape_markdown``),
# which it never applies inside tables.
_ESCAPE = r"\\[\\`*_{}\[\]()#+\-.!]"
_ESCAPE_TOKEN = re.compile(_ESCAPE)
_ESCAPE_OR_WHITESPACE_TOKEN = re.compile(_ESCAPE + r"|\s{2,}|[^\S ]")
# The pipe renderer writes every row as "| " + " | ".join(cells) + " |",
# indented when the table sits in a list item.
_ROW_START = re.compile(r"[ \t]*\| ")
_WHITESPACE_RUN = re.compile(r"\s+")


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


class ReaderView(NamedTuple):
    """Rendered text as a reader sees it, with a sparse map back to the rendered text.

    ``anchors`` lists, ascending, the view index of every character whose
    rendered span is longer than one character; ``extra`` holds the running
    total of those extra rendered characters through each anchor.
    """

    text: str
    anchors: list[int]
    extra: list[int]

    def to_rendered(self, view_offset: int) -> int:
        """The rendered offset of a view boundary (the start of the span at ``view_offset``)."""
        k = bisect_left(self.anchors, view_offset)
        return view_offset + (self.extra[k - 1] if k else 0)


def _table_rows(text: str) -> tuple[list[int], list[int]]:
    """Start and end offsets of every rendered table row, in order.

    ``MarkdownRenderer._render_table_pipe`` (the format ``Attachment.markdown()``
    uses) writes each row as ``"| " + " | ".join(cells) + " |"`` with stripped
    cells, which may contain newlines and even blank lines. A row therefore
    starts on a line matching ``[ \\t]*\\| `` and closes on the first line that
    ends with `` |``; its continuation lines never start with ``| `` and never
    end with `` |`` while cells are pipe-free. A candidate that does not close
    before the next row start or the end of text is prose, so the reviewer's
    ``|x| = 5 ...`` line and prose that merely starts with ``| `` stay
    escaped. Unclassified text defaults to prose on purpose: calling prose a
    table would leave its escapes in place (a common false negative), while
    calling a table prose only mis-reads a genuine backslash followed by an
    escapable character in a cell (rare in SEC tables).
    """
    starts: list[int] = []
    ends: list[int] = []
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    index = 0
    while index < len(lines):
        if not _ROW_START.match(lines[index]):
            index += 1
            continue
        close = index
        while not lines[close].endswith(" |"):
            close += 1
            if close == len(lines) or _ROW_START.match(lines[close]):
                close = None
                break
        if close is None:
            index += 1
            continue
        starts.append(offsets[index])
        ends.append(offsets[close] + len(lines[close]))
        index = close + 1
    return starts, ends


def _in_table_row(position: int, starts: list[int], ends: list[int]) -> bool:
    k = bisect_right(starts, position) - 1
    return k >= 0 and position < ends[k]


def reader_view(text: str, collapse_whitespace: bool) -> ReaderView:
    """The reader view of rendered ``text`` (see the module docstring)."""
    token = _ESCAPE_OR_WHITESPACE_TOKEN if collapse_whitespace else _ESCAPE_TOKEN
    starts, ends = _table_rows(text)
    parts: list[str] = []
    anchors: list[int] = []
    extra: list[int] = []
    view_length = 0
    last = 0
    for match in token.finditer(text):
        span = match.group()
        if span[0] == "\\":
            if _in_table_row(match.start(), starts, ends):
                continue  # the renderer never escapes tables: this backslash is text
            replacement = span[1]
        else:
            replacement = " "
        parts.append(text[last:match.start()])
        view_length += match.start() - last
        parts.append(replacement)
        if len(span) > 1:
            anchors.append(view_length)
            extra.append((extra[-1] if extra else 0) + len(span) - 1)
        view_length += 1
        last = match.end()
    if not parts:
        return ReaderView(text, anchors, extra)
    parts.append(text[last:])
    return ReaderView("".join(parts), anchors, extra)


def _context(text: str, start: int, end: int) -> str:
    ctx_start = max(0, start - CONTEXT_CHARS)
    ctx_end = min(len(text), end + CONTEXT_CHARS)
    context = text[ctx_start:ctx_end].strip()
    if ctx_start > 0:
        context = "..." + context
    if ctx_end < len(text):
        context = context + "..."
    return context


def rendered_matches(text: str, query: str, regex: bool, filename: str, location: str, remaining: int) -> list:
    """Match records for one document's rendered ``text``, matched through its reader view."""
    view = reader_view(text, collapse_whitespace=not regex)
    pattern = query if regex else _WHITESPACE_RUN.sub(" ", query)
    matches = _grep_text(
        view.text,
        pattern,
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
    records = []
    for match in matches:
        start = view.to_rendered(match.char_offset)
        end = view.to_rendered(match.char_offset + len(match.match))
        records.append({
            "location": match.location,
            "match": text[start:end],
            "match_text": match.match,
            "context": _context(text, start, end),
            "locator": {"document": filename, "char_offset": start},
        })
    return records


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
        records.extend(rendered_matches(text, query, regex, filename, _location(attachment), remaining))
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

