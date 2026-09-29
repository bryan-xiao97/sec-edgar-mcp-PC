"""Fast contract tests for the Phase 2 QA fixes to ``edgar_document`` and its handoffs.

Covers findings P2-H1, M1-M3 and L1-L16 in ``qa-phase2-report.md`` plus
cursor-only continuation (constraints rule 7b). All SEC lookup is
monkeypatched; the fakes are shared with ``tests/test_mcp_document.py``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import edgar
from edgar._filings import Filing
from edgar.ai.mcp.tools import continuation
from edgar.ai.mcp.tools.continuation import ResultCache
from tests.test_mcp_document import ACCESSION, FakeAttachment, make_filing

URL_BASE = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000073"
ROUTE_TEXT = "read the exhibit-index section or use edgar_text_search"


@pytest.fixture(autouse=True)
def isolate_mcp_caches(monkeypatch):
    monkeypatch.setattr(continuation, "text_cache", ResultCache(max_entries=64, max_bytes=4_000_000))
    monkeypatch.setattr(continuation, "results_cache", ResultCache(max_entries=16))


@pytest.fixture
def filing(monkeypatch):
    """A 10-Q with readable exhibits, unreadable exhibits and XBRL viewer noise."""
    attachments = [
        FakeAttachment("1", "apple.htm", "10-Q", "cover text needle", description="Quarterly report"),
        FakeAttachment("4", "a10-qexhibit32106282025.htm", "EX-32.1", "CEO Timothy D. Cook"),
        FakeAttachment("5", "agreement-a.htm", "EX-10.1", "needle one\nneedle two"),
        FakeAttachment("6", "empty.htm", "EX-10.2", "", markdown=""),
        FakeAttachment("7", "scan.pdf", "EX-99.1", None),
        FakeAttachment("8", "broken.htm", "EX-99.2", None, unreadable_error="renderer exploded"),
        FakeAttachment("20", "R2.htm", "XML", "generated needle"),
        FakeAttachment("56", "Show.js", "JS", "needle in viewer script"),
        FakeAttachment("57", "report.css", "CSS", "needle css"),
        FakeAttachment("59", "FilingSummary.xml", "XML", "summary"),
        FakeAttachment("62", "MetaLinks.json", "JSON", "needle json"),
        FakeAttachment("63", "0000320193-25-000073-xbrl.zip", "ZIP", None),
        FakeAttachment("64", "Financial_Report.xlsx", "XLSX", None),
    ]
    fake = make_filing(attachments)
    monkeypatch.setattr(Filing, "attachments", property(lambda self: self._test_attachments))
    monkeypatch.setattr("edgar.find", lambda **kwargs: fake)
    return fake


def _decode(cursor: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))


async def _call(**kwargs):
    from edgar.ai.mcp.tools.document import edgar_document

    return await edgar_document(**kwargs)


# ---------------------------------------------------------------------------
# P2-H1: search never reports zero matches for content it could not read
# ---------------------------------------------------------------------------

@pytest.mark.fast
@pytest.mark.parametrize(
    ("selector", "filename", "reason_fragment"),
    [
        ("EX-99.1", "scan.pdf", "pdf"),
        ("EX-10.2", "empty.htm", "did not produce readable text"),
        ("EX-99.2", "broken.htm", "renderer exploded"),
    ],
)
async def test_search_of_unreadable_selected_document_returns_unreadable_reason(
    filing, selector, filename, reason_fragment
):
    result = await _call(action="search", accession_number=ACCESSION, document=selector, query="guaranty")

    assert result.success is True, result.error
    assert result.data["document"]["filename"] == filename
    assert result.data["document"]["url"].endswith(filename)
    assert reason_fragment in result.data["unreadable_reason"].lower()
    assert result.data["matches"] is None
    assert result.data["page"] is None
    assert result.data["source"]["document_url"].endswith(filename)


@pytest.mark.fast
async def test_all_documents_search_lists_every_unsearched_document(filing):
    filing._test_attachments.append(FakeAttachment("30", "", "EX-99.9", None))
    result = await _call(action="search", accession_number=ACCESSION, query="no such phrase")

    assert result.success is True, result.error
    assert result.data["matches"] == []
    unsearched = result.data["unsearched_documents"]
    assert [(u["document"], u["sequence"]) for u in unsearched] == [
        ("empty.htm", "6"), ("scan.pdf", "7"), ("broken.htm", "8"), (None, "30"),
    ]
    assert "pdf" in unsearched[1]["reason"].lower()
    assert "renderer exploded" in unsearched[2]["reason"]
    assert "4 documents" in result.data["coverage_note"]
    assert "zero" in result.data["coverage_note"].lower()


@pytest.mark.fast
async def test_all_documents_search_includes_paper_attachment_as_unsearched(filing):
    filing._test_attachments.append(FakeAttachment("31", "submission.paper", "TEXT", "paper needle"))
    result = await _call(action="search", accession_number=ACCESSION, query="needle")

    paper = [u for u in result.data["unsearched_documents"] if u["document"] == "submission.paper"]
    assert paper and "paper" in paper[0]["reason"].lower()
    assert "submission.paper" not in {m["locator"]["document"] for m in result.data["matches"]}


@pytest.mark.fast
async def test_readable_selected_search_reports_empty_unsearched_list(filing):
    result = await _call(action="search", accession_number=ACCESSION, document="EX-10.1", query="needle")

    assert [m["locator"]["char_offset"] for m in result.data["matches"]] == [0, 11]
    assert result.data["unsearched_documents"] == []
    assert "coverage_note" not in result.data


# ---------------------------------------------------------------------------
# P2-M1 / P2-L16: documented URL example and selected_by="url"
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_documented_url_example_names_the_june_2025_exhibit():
    from edgar.ai.mcp.server import _import_tools
    from edgar.ai.mcp.tools.base import TOOLS

    _import_tools()
    description = TOOLS["edgar_document"]["description"]
    quickstart = (Path(edgar.__file__).parent / "ai/mcp/docs/MCP_QUICKSTART.md").read_text()
    good = f"{URL_BASE}/a10-qexhibit32106282025.htm"

    assert good in description and good in quickstart
    assert "a10-qexhibit32103292025.htm" not in description + quickstart
    assert 'selected_by: "url"' in description


# ---------------------------------------------------------------------------
# P2-M2: source.period_of_report
# ---------------------------------------------------------------------------

@pytest.mark.fast
async def test_period_of_report_comes_from_loaded_sgml_header(filing):
    filing._sgml = SimpleNamespace(period_of_report="2025-06-28")
    result = await _call(action="list", accession_number=ACCESSION)

    assert result.data["source"]["period_of_report"] == "2025-06-28"


@pytest.mark.fast
async def test_period_of_report_failure_serializes_null_not_error(filing):
    class BrokenHeader:
        @property
        def period_of_report(self):
            raise OSError("header unavailable")

    filing._sgml = BrokenHeader()
    result = await _call(action="list", accession_number=ACCESSION)

    assert result.success is True, result.error
    assert result.data["source"]["period_of_report"] is None


# ---------------------------------------------------------------------------
# P2-M3: XBRL viewer support files are hidden by default
# ---------------------------------------------------------------------------

@pytest.mark.fast
async def test_default_list_hides_xbrl_viewer_support_files(filing):
    default = await _call(action="list", accession_number=ACCESSION, limit=50)
    everything = await _call(action="list", accession_number=ACCESSION, include_all=True, limit=50)

    assert [d["filename"] for d in default.data["documents"]] == [
        "apple.htm", "a10-qexhibit32106282025.htm", "agreement-a.htm", "empty.htm", "scan.pdf", "broken.htm",
    ]
    assert {
        "Show.js", "report.css", "FilingSummary.xml", "MetaLinks.json",
        "0000320193-25-000073-xbrl.zip", "Financial_Report.xlsx", "R2.htm",
    }.issubset({d["filename"] for d in everything.data["documents"]})


@pytest.mark.fast
async def test_all_documents_search_skips_viewer_files(filing):
    result = await _call(action="search", accession_number=ACCESSION, query="needle")

    assert {m["locator"]["document"] for m in result.data["matches"]} == {"apple.htm", "agreement-a.htm"}


# ---------------------------------------------------------------------------
# P2-L1 / P2-L2: URL identity is exact
# ---------------------------------------------------------------------------

@pytest.mark.fast
@pytest.mark.parametrize("name", ["10-Q", "EX-10", "5", "ex-10.1", "primary", "AGREEMENT-A.HTM"])
async def test_url_filename_resolves_by_exact_filename_only(filing, name):
    result = await _call(action="read", url=f"{URL_BASE}/{name}")

    assert result.success is False
    assert result.error_code == "DOCUMENT_NOT_FOUND"


@pytest.mark.fast
async def test_url_cik_must_be_a_filer_of_the_accession(filing):
    filing._related_entities = [{"cik": 1234, "company": "Co-Registrant LLC"}]
    wrong = await _call(action="read", url="https://www.sec.gov/Archives/edgar/data/999999/000032019325000073/agreement-a.htm")
    related = await _call(action="read", url="https://www.sec.gov/Archives/edgar/data/1234/000032019325000073/agreement-a.htm")
    padded = await _call(action="read", url="https://www.sec.gov/Archives/edgar/data/0000320193/000032019325000073/agreement-a.htm")

    assert wrong.success is False
    assert wrong.error_code == "FILING_MISMATCH"
    assert "999999" in wrong.error
    assert related.success is True, related.error
    assert padded.success is True, padded.error


@pytest.mark.fast
@pytest.mark.parametrize("path", ["sub/agreement-a.htm", "a/b/agreement-a.htm", "sub%2Fagreement-a.htm"])
async def test_url_subdirectory_paths_are_rejected(monkeypatch, path):
    monkeypatch.setattr("edgar.find", lambda **kwargs: pytest.fail("rejected before lookup"))
    result = await _call(action="read", url=f"{URL_BASE}/{path}")

    assert result.error_code == "INVALID_URL"


@pytest.mark.fast
@pytest.mark.parametrize(
    "name", ["0000320193-25-000073-index.htm", "0000320193-25-000073-index-headers.html", "0000320193-25-000073.txt", "index.html"]
)
async def test_edgar_document_treats_index_urls_as_filing_level(filing, name):
    listed = await _call(action="list", url=f"{URL_BASE}/{name}")
    read = await _call(action="read", url=f"{URL_BASE}/{name}")

    assert listed.success is True, listed.error
    assert read.error_code == "DOCUMENT_REQUIRED"


# ---------------------------------------------------------------------------
# P2-L5 / P2-L6: regex engine parity and hashed query cursors
# ---------------------------------------------------------------------------

@pytest.mark.fast
async def test_regex_validation_uses_the_search_engine(filing):
    unicode_class = await _call(action="search", accession_number=ACCESSION, document="EX-32.1", query=r"\p{Lu}\. Cook", regex=True)
    invalid = await _call(action="search", accession_number=ACCESSION, query=r"[unclosed", regex=True)

    assert unicode_class.success is True, unicode_class.error
    assert [m["match"] for m in unicode_class.data["matches"]] == ["D. Cook"]
    assert invalid.error_code == "INVALID_QUERY"


@pytest.mark.fast
def test_timed_grep_surfaces_regex_compile_errors():
    from edgar.exceptions import ValidationError
    from edgar.search.grep import _grep_text

    with pytest.raises(ValidationError, match="Invalid regular expression"):
        _grep_text("some text", r"[unclosed", "doc1", regex=True, regex_timeout=0.05)
    # The untimed public path keeps its historical behaviour.
    assert _grep_text("some text", r"[unclosed", "doc1", regex=True) == []


@pytest.mark.fast
async def test_long_query_first_page_succeeds_and_cursor_carries_a_hash(filing):
    query = "needle|" + "|".join(f"absent-term-{i}" for i in range(300))
    assert len(query) > 4_000
    first = await _call(action="search", accession_number=ACCESSION, document="EX-10.1", query=query, regex=True, limit=1)

    assert first.success is True, first.error
    cursor = first.data["page"]["next_cursor"]
    assert len(cursor) <= 2_048
    q = _decode(cursor)["q"]
    assert "query" not in q and len(q["query_sha256"]) == 64
    second = await _call(action="search", accession_number=ACCESSION, document="EX-10.1", query=query, regex=True, cursor=cursor, limit=1)
    assert [m["locator"]["char_offset"] for m in second.data["matches"]] == [11]


# ---------------------------------------------------------------------------
# P2-L11 / P2-L12: IBR route, filer on candidates, list arguments, primary
# ---------------------------------------------------------------------------

@pytest.mark.fast
async def test_incorporated_by_reference_route_on_list_search_and_read(filing):
    listed = await _call(action="list", accession_number=ACCESSION)
    searched = await _call(action="search", accession_number=ACCESSION, document="EX-10.1", query="needle")
    read = await _call(action="read", accession_number=ACCESSION, document="EX-10.1")
    unreadable = await _call(action="read", accession_number=ACCESSION, document="EX-99.1")

    for response in (listed, searched, read, unreadable):
        assert ROUTE_TEXT in response.data["incorporated_by_reference_note"]


@pytest.mark.fast
async def test_ambiguous_candidates_name_the_filer(filing):
    filing._test_attachments.append(FakeAttachment("40", "agreement-z.htm", "EX-10.3", "z"))
    result = await _call(action="read", accession_number=ACCESSION, document="EX-10")

    assert result.error_code == "AMBIGUOUS_DOCUMENT"
    assert result.data["filer"] == {"cik": 320193, "name": "Apple Inc."}
    assert {c["filename"] for c in result.data["candidates"]} == {"agreement-a.htm", "empty.htm", "agreement-z.htm"}


@pytest.mark.fast
@pytest.mark.parametrize(
    "extra", [{"document": "EX-10.1"}, {"around": {"document": "agreement-a.htm", "char_offset": 0}}]
)
async def test_list_rejects_document_and_around(filing, extra):
    result = await _call(action="list", accession_number=ACCESSION, **extra)

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"


@pytest.mark.fast
async def test_primary_selector_is_sequence_one(filing):
    result = await _call(action="read", accession_number=ACCESSION, document="primary")

    assert result.success is True, result.error
    assert result.data["document"]["filename"] == "apple.htm"
    assert result.data["text"] == "cover text needle"


# ---------------------------------------------------------------------------
# Cursor-only continuation (constraints rule 7b)
# ---------------------------------------------------------------------------

def _long_text(attachment, marker="needle"):
    text = "".join(f"{marker} {i:05d} " + "x" * 90 + "\n" for i in range(200))
    attachment._text = text
    attachment._markdown = text
    return text


@pytest.mark.fast
async def test_read_cursor_alone_reassembles_the_document(filing):
    text = _long_text(filing._test_attachments[2])
    page = await _call(action="read", accession_number=ACCESSION, document="EX-10.1")
    pages = [page.data["text"]]
    while page.data["page"]["next_cursor"]:
        page = await _call(action="read", cursor=page.data["page"]["next_cursor"])
        assert page.success is True, page.error
        assert page.data["document"]["filename"] == "agreement-a.htm"
        assert page.data["source"]["accession_number"] == ACCESSION
        pages.append(page.data["text"])
    assert "".join(pages) == text
    assert len(pages) == 4


@pytest.mark.fast
async def test_search_cursor_alone_continues_query_and_document(filing):
    _long_text(filing._test_attachments[2])
    first = await _call(action="search", accession_number=ACCESSION, document="agreement-a.htm", query="needle", limit=50)
    second = await _call(action="search", cursor=first.data["page"]["next_cursor"], limit=50)

    assert second.success is True, second.error
    assert second.data["page"]["offset"] == 50
    assert {m["locator"]["document"] for m in second.data["matches"]} == {"agreement-a.htm"}
    assert second.data["matches"][0]["match"].lower() == "needle"


@pytest.mark.fast
async def test_list_cursor_alone_keeps_include_all(filing):
    first = await _call(action="list", accession_number=ACCESSION, include_all=True, limit=5)
    second = await _call(action="list", cursor=first.data["page"]["next_cursor"], limit=5)

    assert second.success is True, second.error
    assert second.data["page"]["total"] == 13
    assert [d["filename"] for d in second.data["documents"]][0] == "broken.htm"


@pytest.mark.fast
async def test_cursor_with_differing_arguments_is_cursor_mismatch(filing):
    _long_text(filing._test_attachments[2])
    read = await _call(action="read", accession_number=ACCESSION, document="EX-10.1")
    search = await _call(action="search", accession_number=ACCESSION, document="EX-10.1", query="needle", limit=1)
    read_cursor = read.data["page"]["next_cursor"]
    search_cursor = search.data["page"]["next_cursor"]

    other_accession = await _call(action="read", accession_number="0000320193-25-000099", cursor=read_cursor)
    other_document = await _call(action="read", document="EX-32.1", cursor=read_cursor)
    other_query = await _call(action="search", query="one", cursor=search_cursor)
    other_regex = await _call(action="search", regex=True, cursor=search_cursor)
    other_action = await _call(action="search", query="needle", cursor=read_cursor)
    same_selector = await _call(action="read", accession_number="000032019325000073", document="5", cursor=read_cursor)

    assert other_accession.error_code == "CURSOR_MISMATCH"
    assert other_document.error_code == "CURSOR_MISMATCH"
    assert other_query.error_code == "CURSOR_MISMATCH"
    assert other_regex.error_code == "CURSOR_MISMATCH"
    assert other_action.error_code == "CURSOR_MISMATCH"
    assert same_selector.success is True, same_selector.error


@pytest.mark.fast
async def test_read_cursor_alone_continues_a_hidden_document_read_with_include_all(filing):
    _long_text(filing._test_attachments[6])  # R2.htm, hidden by default
    first = await _call(action="read", accession_number=ACCESSION, document="R2.htm", include_all=True)
    second = await _call(action="read", cursor=first.data["page"]["next_cursor"])

    assert second.success is True, second.error
    assert second.data["document"]["filename"] == "R2.htm"
    assert second.data["page"]["offset"] == len(first.data["text"])


@pytest.mark.fast
async def test_garbage_cursor_alone_is_invalid_cursor(filing):
    result = await _call(action="read", cursor="not-a-cursor")

    assert result.error_code == "INVALID_CURSOR"


@pytest.mark.fast
async def test_long_query_cursor_alone_asks_for_the_query(filing):
    query = "needle|" + "|".join(f"absent-term-{i}" for i in range(300))
    first = await _call(action="search", accession_number=ACCESSION, document="EX-10.1", query=query, regex=True, limit=1)
    result = await _call(action="search", cursor=first.data["page"]["next_cursor"])

    assert result.success is False
    assert result.error_code == "QUERY_REQUIRED"
    assert "hash" in result.error.lower()


# ---------------------------------------------------------------------------
# edgar_filing and edgar_text_search handoffs (P2-L3, L4, L7, L13)
# ---------------------------------------------------------------------------

class _HandoffFiling:
    accession_no = "0000320193-23-000077"
    form = "10-Q"
    company = "Apple Inc."
    filing_date = "2023-05-05"
    report_date = "2023-04-01"
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/index.html"

    def __init__(self, attachments):
        self.attachments = attachments

    def obj(self):
        return None

    def to_context(self, detail="standard"):
        return "context"


@pytest.mark.fast
@pytest.mark.parametrize(
    "tail",
    [
        "0000320193-23-000077-index.htm",
        "0000320193-23-000077-index.html",
        "0000320193-23-000077-index-headers.html",
        "0000320193-23-000077.txt",
        "index.html",
        "",
    ],
)
async def test_edgar_filing_index_urls_are_filing_level(monkeypatch, tail):
    from edgar.ai.mcp.tools.filing import edgar_filing

    monkeypatch.setattr(edgar, "find", lambda *, search_id: _HandoffFiling([]))
    response = await edgar_filing(input=f"https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/{tail}")

    assert response.success is True, response.error
    assert "document_hint" not in response.data
    assert not any("not found" in step.lower() for step in response.next_steps)


@pytest.mark.fast
async def test_edgar_filing_hint_for_hidden_document_sets_include_all(monkeypatch):
    from edgar.ai.mcp.tools.filing import edgar_filing

    hidden = SimpleNamespace(document="R2.htm", sequence_number="20", document_type="XML", ixbrl=False)
    visible = SimpleNamespace(document="credit.htm", sequence_number="7", document_type="EX-10.1", ixbrl=False)
    monkeypatch.setattr(edgar, "find", lambda *, search_id: _HandoffFiling([hidden, visible]))
    base = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000077"

    hidden_response = await edgar_filing(input=f"{base}/R2.htm")
    visible_response = await edgar_filing(input=f"{base}/credit.htm")

    hidden_step = next(s for s in hidden_response.next_steps if "edgar_document" in s)
    visible_step = next(s for s in visible_response.next_steps if "edgar_document" in s)
    assert '"include_all": true' in hidden_step and '"document": "R2.htm"' in hidden_step
    assert "include_all" not in visible_step


@pytest.mark.fast
@pytest.mark.parametrize("tool_name", ["edgar_filing", "edgar_text_search"])
def test_marked_json_block_follows_the_final_example_line(tool_name):
    from edgar.ai.mcp.server import _import_tools
    from edgar.ai.mcp.tools.base import TOOLS

    _import_tools()
    description = TOOLS[tool_name]["description"]
    marker = description.index("<!-- MCP_TOOL_CALL_EXAMPLE -->")
    examples = description[:marker]
    assert description.rstrip().endswith("```")
    last_example = {"edgar_filing": "- Minimal overview:", "edgar_text_search": "- Company-specific:"}[tool_name]
    assert last_example in examples
    assert "\n- " not in description[marker:]


@pytest.mark.fast
async def test_text_search_caps_per_hit_steps_at_five_unique_accessions(monkeypatch):
    from edgar.ai.mcp.tools.text_search import edgar_text_search
    from edgar.search.efts import EFTSResult, EFTSSearch

    hits = [
        EFTSResult(accession_number=f"0000320193-23-00000{i // 2}", form="8-K", filed="2023-05-05",
                   document_id=f"ex{i}.htm")
        for i in range(16)
    ]
    monkeypatch.setattr(
        "edgar.search.efts.search_filings",
        lambda *a, **k: EFTSSearch(query="credit agreement", total=16, results=hits),
    )
    response = await edgar_text_search(query="credit agreement")

    per_hit = [s for s in response.next_steps if s.startswith("For ")]
    accessions = [s.split()[1].rstrip(",") for s in per_hit]
    assert accessions == [f"0000320193-23-00000{i}" for i in range(5)]
    assert "ex0.htm" in per_hit[0] and "ex1.htm" not in per_hit[0]
    assert len(response.data["results"]) == 16
