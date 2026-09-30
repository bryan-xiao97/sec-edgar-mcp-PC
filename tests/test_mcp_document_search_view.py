"""Search over the reader's view of rendered markdown (Q4 fix rounds 1-3).

Literal queries match the text a reader sees: markdown escapes are removed
from prose (never from table rows, which the renderer does not escape), and
any run of whitespace matches any other run. Regex queries keep the user's
own semantics over the unescaped text. Every locator still indexes the exact
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

# What edgar.documents' markdown renderer escapes in prose (``_escape_markdown``).
_ESCAPABLE = "\\`*_{}[]()#+-.!"


def _markdown(html: str) -> str:
    """Render HTML through the real edgar.documents pipeline Attachment.markdown() uses."""
    from edgar.documents import HTMLParser, ParserConfig

    return HTMLParser(ParserConfig()).parse(html).to_markdown()


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


# A document mixing prose, a headed table, a prose line that starts with "|",
# a header-less table and a list-item table, rendered by the real renderer.
_MIXED_HTML = (
    "<html><body><p>Before the table, Section 15(d) of the U.S. Code applies.</p>"
    "<table><tr><th>Metric</th><th>Value</th></tr><tr><td>Yield</td><td>5.5%\\*</td></tr>"
    "<tr><td>Net asset value (per share)</td><td>$1,000.00</td></tr></table>"
    "<p>|x| = 5 for clause 15(d) here, first-lien.</p>"
    "<table><tr><td>Rate \\(floor\\)</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    "<ul><li>Item (1)<table><tr><th>H</th></tr><tr><td>x\\.y</td></tr></table></li></ul>"
    "<p>After the tables, Sarbanes-Oxley applies.</p></body></html>"
)


@pytest.mark.fast
def test_prose_line_starting_with_a_pipe_is_still_unescaped():
    """The reviewer's repro: a leading "|" alone does not make a table row."""
    rendered = "|x| = 5 for clause 15\\(d\\) here"
    records = rendered_matches(rendered, "15(d)", False, "doc.htm", "doc", remaining=100)

    assert [(r["locator"]["char_offset"], r["match"]) for r in records] == [(19, "15\\(d\\)")]


@pytest.mark.fast
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("15(d)", ["15\\(d\\)", "15\\(d\\)"]),  # prose before the table, and the "|x|" prose after it
        ("U.S.", ["U\\.S\\."]),
        ("first-lien", ["first\\-lien"]),
        ("Sarbanes-Oxley", ["Sarbanes\\-Oxley"]),  # prose after all three tables
        ("5.5%\\*", ["5.5%\\*"]),  # literal backslash in a headed table
        ("net asset value (per share)", ["Net asset value (per share)"]),  # table cells are not escaped
        ("Rate \\(floor\\)", ["Rate \\(floor\\)"]),  # literal backslashes in a header-less table
        ("x\\.y", ["x\\.y"]),  # literal backslash in an indented list-item table
        ("Item (1)", ["Item \\(1\\)"]),
    ],
)
async def test_real_rendered_tables_and_prose_are_classified_correctly(filing, query, expected):  # noqa: F811
    rendered = _markdown(_MIXED_HTML)
    assert "| --- | --- |" in rendered and "\n\n|x| = 5" in rendered and "\n  | x\\.y |" in rendered
    filing._test_attachments[2]._markdown = rendered
    searched = await _search(query)

    assert [m["match"] for m in searched.data["matches"]] == expected
    _assert_exact(searched.data["matches"], rendered)


_PROSE_ALPHABET = [
    "a", "b", "İ", "ı", "σ", "ς", "Σ", "e\u0307", "(", ")", "-", ".", "\\", "*", "%",
    " ", "  ", "\t", "\u00a0", "|", "| ", "x",
]
_CELL_ALPHABET = ["a", "İ", "σ", "(", ")", "-", ".", "\\", "*", "%", " ", "\u00a0", "\n", "\n\n", "x", "$"]


def _prose_block(rng) -> list[tuple[str, str]]:
    """(original, rendered) pieces of escaped prose; lines may start with "|" or "| "."""
    lines = []
    for _ in range(rng.randint(1, 3)):
        line = "".join(rng.choice(_PROSE_ALPHABET) for _ in range(rng.randint(1, 30)))
        # Documented ambiguity: a prose line ending in " |" could close a
        # "row" opened by an earlier prose line starting with "| ".
        lines.append(line + "x" if line.endswith(" |") else line)
    pieces = []
    for index, line in enumerate(lines):
        if index:
            pieces.append(("\n", "\n"))
        pieces.extend((ch, "\\" + ch if ch in _ESCAPABLE else ch) for ch in line)
    return pieces


def _table_block(rng) -> list[tuple[str, str]]:
    """Pieces of a pipe table exactly as the renderer builds one: never escaped."""
    columns = rng.randint(1, 3)

    def cell() -> str:  # the renderer strips cells; real cells may span blank lines
        return "".join(rng.choice(_CELL_ALPHABET) for _ in range(rng.randint(0, 8))).strip()

    rows = []
    if rng.random() < 0.5:
        rows += [[cell() for _ in range(columns)], ["---"] * columns]
    rows += [[cell() for _ in range(columns)] for _ in range(rng.randint(1, 3))]
    text = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    if rng.random() < 0.25:  # a table inside a list item: every line indented
        text = "\n".join("  " + line for line in text.split("\n"))
    return [(ch, ch) for ch in text]


@pytest.mark.fast
def test_view_offsets_are_exact_under_unicode_whitespace_and_table_fuzz():
    """Generated documents carry ground truth: which text is table, which is escaped prose."""
    rng = random.Random(20260929)  # noqa: S311 -- deterministic fuzz seed, not cryptography
    checked = 0
    for case in range(3_000):
        pieces: list[tuple[str, str]] = []
        for index in range(rng.randint(1, 4)):
            if index:
                pieces.extend([("\n", "\n"), ("\n", "\n")])  # one piece per character
            pieces.extend(_table_block(rng) if rng.random() < 0.4 else _prose_block(rng))
        original = "".join(o for o, _ in pieces)
        rendered = "".join(r for _, r in pieces)
        to_original = {0: 0}
        rendered_length = original_length = 0
        for o, r in pieces:
            rendered_length += len(r)
            original_length += len(o)
            to_original[rendered_length] = original_length

        start = rng.randrange(len(original))
        raw_query = original[start:start + rng.randint(1, 8)]
        regex = case % 2 == 1
        if regex:
            query = re.escape(raw_query)
            oracle = _grep_text(original, query, "doc", regex=True, regex_timeout=1.0)
        else:
            query = raw_query
            oracle = _grep_text(re.sub(r"\s+", " ", original), re.sub(r"\s+", " ", query), "doc")
        records = rendered_matches(rendered, query, regex, "doc.htm", "doc", remaining=10_000)
        assert len(records) == len(oracle), (rendered, query)
        for record, expected in zip(records, oracle, strict=True):
            offset = record["locator"]["char_offset"]
            assert rendered[offset:offset + len(record["match"])] == record["match"]
            prefix = original[:to_original[offset]]
            assert len(prefix if regex else re.sub(r"\s+", " ", prefix)) == expected.char_offset
            assert record["match_text"] == expected.match
            checked += 1
    assert checked > 3_000
