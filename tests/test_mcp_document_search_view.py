"""Search over the reader's view of rendered markdown (Q4 fix rounds 1 and 2).

Literal queries match the text a reader sees: markdown escapes are removed
(outside table rows, which the renderer never escapes), and any run of
whitespace matches any other run. Regex queries keep the user's own
semantics over the unescaped text. Every locator still indexes the exact
rendered text ``read`` pages.
"""
from __future__ import annotations

import random
import re

import pytest

from edgar.ai.mcp.tools.document_search import rendered_matches
from edgar.search.grep import _grep_text
from tests.test_mcp_document import ACCESSION
from tests.test_mcp_document_contract import filing, isolate_mcp_caches  # noqa: F401  (fixtures)

# What edgar.documents' markdown renderer escapes, and only outside tables.
_ESCAPABLE = "\\`*_{}[]()#+-.!"
_TABLE_ROW = re.compile(r"[ \t]*\|")


def _render(original: str) -> str:
    """Mimic the renderer: escape non-table lines, leave table rows verbatim."""
    return "\n".join(
        line if _TABLE_ROW.match(line) else "".join("\\" + ch if ch in _ESCAPABLE else ch for ch in line)
        for line in original.split("\n")
    )


def _reference_view(rendered: str, collapse: bool) -> str:
    """An independent, line-based reader view used as the test oracle."""
    lines = [
        line if _TABLE_ROW.match(line) else re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", r"\1", line)
        for line in rendered.split("\n")
    ]
    text = "\n".join(lines)
    return re.sub(r"\s+", " ", text) if collapse else text


async def _call(**kwargs):
    from edgar.ai.mcp.tools.document import edgar_document

    return await edgar_document(**kwargs)


_WRAPPED = (
    "Net\nasset value per share rose. The net   asset value fell. Net\tasset value held.\n"
    "| Net asset   value | 1.00 |\n"
    "| Net asset | value |\n"
    "Net asset value\\."
)


@pytest.fixture
def wrapped_exhibit(filing):  # noqa: F811
    exhibit = filing._test_attachments[2]  # agreement-a.htm, EX-10.1
    exhibit._markdown = _WRAPPED
    return exhibit


async def _search(query: str, regex: bool = False):
    return await _call(action="search", accession_number=ACCESSION, document="EX-10.1", query=query, regex=regex)


def _assert_exact(matches: list, rendered: str) -> None:
    for m in matches:
        offset = m["locator"]["char_offset"]
        assert rendered[offset:offset + len(m["match"])] == m["match"]


@pytest.mark.fast
@pytest.mark.parametrize("query", ["net asset value", "net \n  asset\tvalue", "NET  ASSET VALUE"])
async def test_literal_whitespace_matches_any_run_and_query_whitespace_collapses(wrapped_exhibit, query):
    searched = await _search(query)
    read = await _call(action="read", accession_number=ACCESSION, document="EX-10.1")

    matches = searched.data["matches"]
    assert read.data["text"] == _WRAPPED
    assert [m["match"] for m in matches] == [
        "Net\nasset value", "net   asset value", "Net\tasset value", "Net asset   value", "Net asset value",
    ]
    assert [m["match_text"] for m in matches] == [
        "Net asset value", "net asset value", "Net asset value", "Net asset value", "Net asset value",
    ]
    _assert_exact(matches, read.data["text"])


@pytest.mark.fast
async def test_literal_match_never_spans_a_table_cell_boundary(wrapped_exhibit):
    searched = await _search("net asset value")

    offsets = [m["locator"]["char_offset"] for m in searched.data["matches"]]
    cross_cell_row = _WRAPPED.index("| Net asset | value |")
    assert not any(cross_cell_row <= o < cross_cell_row + len("| Net asset | value |") for o in offsets)


@pytest.mark.fast
async def test_regex_keeps_its_own_whitespace_semantics(wrapped_exhibit):
    exact_space = await _search("net asset value", regex=True)
    tolerant = await _search(r"net\s+asset\s+value", regex=True)

    assert [m["match"] for m in exact_space.data["matches"]] == ["Net asset value"]
    assert [m["match"] for m in tolerant.data["matches"]] == [
        "Net\nasset value", "net   asset value", "Net\tasset value", "Net asset   value", "Net asset value",
    ]
    assert tolerant.data["matches"][0]["match_text"] == "Net\nasset value"
    _assert_exact(tolerant.data["matches"], _WRAPPED)


_TABLE_BACKSLASH = "| Yield | 5.5%\\* | see note |\n\nBody yield 5\\.5%\\* and rate 7\\%\\."


@pytest.mark.fast
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("5.5%\\*", ["5.5%\\*"]),  # a genuine backslash inside a table cell (never escaped)
        ("5.5%*", ["5\\.5%\\*"]),  # escaped body text
        ("7\\%", ["7\\%"]),  # "%" is not in the renderer's escape set, so the backslash is real
    ],
)
async def test_unescape_follows_what_the_renderer_escapes(filing, query, expected):  # noqa: F811
    filing._test_attachments[2]._markdown = _TABLE_BACKSLASH
    searched = await _search(query)

    assert [m["match"] for m in searched.data["matches"]] == expected
    _assert_exact(searched.data["matches"], _TABLE_BACKSLASH)


@pytest.mark.fast
def test_view_offsets_are_exact_under_unicode_whitespace_and_table_fuzz():
    alphabet = [
        "a", "b", "İ", "ı", "σ", "ς", "Σ", "ė", "(", ")", "-", ".", "\\", "*", "%",
        " ", "  ", "\t", " ", "\n", "|", "x",
    ]
    rng = random.Random(20260929)  # noqa: S311 -- deterministic fuzz seed, not cryptography
    checked = 0
    for case in range(3_000):
        original = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        rendered = _render(original)
        assert _reference_view(rendered, collapse=False) == original
        start = rng.randrange(len(original))
        raw_query = original[start:start + rng.randint(1, 8)]
        regex = case % 2 == 1
        if regex:
            query = re.escape(raw_query)
            view = original
            oracle = _grep_text(view, query, "doc", regex=True, regex_timeout=1.0)
        else:
            query = raw_query
            view = re.sub(r"\s+", " ", original)
            oracle = _grep_text(view, re.sub(r"\s+", " ", query), "doc")
        records = rendered_matches(rendered, query, regex, "doc.htm", "doc", remaining=10_000)
        assert len(records) == len(oracle), (original, query)
        for record, expected in zip(records, oracle, strict=True):
            offset = record["locator"]["char_offset"]
            assert rendered[offset:offset + len(record["match"])] == record["match"]
            assert len(_reference_view(rendered[:offset], collapse=not regex)) == expected.char_offset
            assert record["match_text"] == expected.match
            checked += 1
    assert checked > 3_000
