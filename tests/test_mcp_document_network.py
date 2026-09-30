"""Ground-truth VCR tests for ``edgar_document`` against real SEC filings (QA P2-L14, P2-M1).

- Princeton Capital Corp 10-Q (CIK 845385, accession 0001213900-26-090000),
  the small-BDC ground-truth filing from Task 3 (constraints rule 7a). Its
  full submission is ~9.6 MB, so the cassette stays under the 10 MB cap.
- Apple 10-Q accession 0000320193-25-000073: the documented URL example in
  the tool description and MCP_QUICKSTART.md must resolve (Verification
  Constitution principle 1).

``edgar.find`` is memoized per test so the full submission is requested
(and recorded) once rather than once per tool call. Values were measured
live on 2026-09-29.
"""
from __future__ import annotations

import pytest

import edgar
from edgar import set_identity
from edgar.ai.mcp.tools import continuation
from edgar.ai.mcp.tools.continuation import ResultCache
from edgar.ai.mcp.tools.document import edgar_document

PRINCETON_ACCESSION = "0001213900-26-090000"
PRINCETON_FOLDER = "https://www.sec.gov/Archives/edgar/data/845385/000121390026090000"
APPLE_EXHIBIT_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/000032019325000073/a10-qexhibit32106282025.htm"
)


@pytest.fixture
def memoized_find(monkeypatch):
    monkeypatch.setattr(continuation, "text_cache", ResultCache(max_entries=64, max_bytes=32_000_000))
    monkeypatch.setattr(continuation, "results_cache", ResultCache(max_entries=16))
    real_find = edgar.find
    loaded = {}

    def find_once(**kwargs):
        key = kwargs["search_id"]
        if key not in loaded:
            loaded[key] = real_find(**kwargs)
        return loaded[key]

    monkeypatch.setattr(edgar, "find", find_once)


@pytest.mark.network
@pytest.mark.vcr
@pytest.mark.asyncio
class TestEdgarDocumentPrincetonVCR:
    async def test_list_read_search_ground_truth(self, memoized_find):
        set_identity("Test User test@test.com")

        listed = await edgar_document(action="list", accession_number=PRINCETON_ACCESSION)
        everything = await edgar_document(action="list", accession_number=PRINCETON_ACCESSION, include_all=True,
                                          limit=50)
        all_filenames = [d["filename"] for d in everything.data["documents"]]
        cursor = everything.data["page"]["next_cursor"]
        while cursor:  # the cursor alone continues the include_all listing (rule 7b)
            page = await edgar_document(action="list", cursor=cursor, limit=50)
            all_filenames += [d["filename"] for d in page.data["documents"]]
            cursor = page.data["page"]["next_cursor"]
        ex32 = await edgar_document(action="read", accession_number=PRINCETON_ACCESSION, document="EX-32")
        ex31 = await edgar_document(action="read", url=f"{PRINCETON_FOLDER}/ea030147301ex31-1.htm")
        searched = await edgar_document(action="search", accession_number=PRINCETON_ACCESSION,
                                        query="Sarbanes-Oxley")

        # list: the 10-Q and its three certifications; viewer noise hidden (P2-M3)
        assert [d["filename"] for d in listed.data["documents"]] == [
            "ea0301473-10q_princeton.htm", "ea030147301ex31-1.htm", "ea030147301ex31-2.htm", "ea030147301ex32.htm",
        ]
        assert len(all_filenames) == everything.data["page"]["total"]
        assert {"Show.js", "report.css", "MetaLinks.json", "0001213900-26-090000-xbrl.zip"}.issubset(all_filenames)
        # provenance (P2-M2)
        source = listed.data["source"]
        assert source["period_of_report"] == "2026-06-30"
        assert (source["cik"], source["entity"], source["filed"]) == (845385, "PRINCETON CAPITAL CORP", "2026-08-14")
        assert source["selected_by"] == "accession"

        # read EX-32 by type and EX-31.1 by URL
        assert ex32.data["document"]["filename"] == "ea030147301ex32.htm"
        assert ex32.data["page"]["total_chars"] == 1292
        assert "| Date: August 14, 2026 | /s/ Mark S. DiSalvo |" in ex32.data["text"]
        assert "Gregory J. Cannella" in ex32.data["text"]
        assert ex31.data["source"]["selected_by"] == "url"
        assert ex31.data["page"]["total_chars"] == 3613
        assert "I, Mark S\\. DiSalvo, certify that:" in ex31.data["text"]

        # search locators index the text read returns
        # Search matches the reader's view of the escaped markdown (Q4 fix I1).
        # The body instance renders as "Sarbanes\\-Oxley"; the heading is unescaped.
        # Counts were verified by hand from the full read text on 2026-09-29.
        assert [(m["locator"]["document"], m["locator"]["char_offset"], m["match"]) for m in searched.data["matches"]] == [
            ("ea0301473-10q_princeton.htm", 162104, "Sarbanes-Oxley"),
            ("ea030147301ex32.htm", 168, "SARBANES-OXLEY"),
            ("ea030147301ex32.htm", 277, "Sarbanes\\-Oxley"),
        ]
        assert searched.data["unsearched_documents"] == []
        for match in searched.data["matches"][1:]:
            offset = match["locator"]["char_offset"]
            assert ex32.data["text"][offset:offset + len(match["match"])] == match["match"]
        assert [m["match_text"] for m in searched.data["matches"]] == [
            "Sarbanes-Oxley", "SARBANES-OXLEY", "Sarbanes-Oxley",
        ]

        section_15d = await edgar_document(action="search", accession_number=PRINCETON_ACCESSION, query="15(d)")
        assert [(m["locator"]["document"], m["locator"]["char_offset"]) for m in section_15d.data["matches"]] == [
            ("ea0301473-10q_princeton.htm", 155),
            ("ea0301473-10q_princeton.htm", 307),
            ("ea0301473-10q_princeton.htm", 1114),
            ("ea0301473-10q_princeton.htm", 162747),
            ("ea030147301ex32.htm", 738),
        ]
        for match in section_15d.data["matches"]:
            assert match["match"] == "15\\(d\\)" and match["match_text"] == "15(d)"
            locator = match["locator"]
            around = await edgar_document(action="read", accession_number=PRINCETON_ACCESSION, around=locator)
            start = locator["char_offset"] - around.data["page"]["offset"]
            assert around.data["text"][start:start + len(match["match"])] == "15\\(d\\)"


@pytest.mark.network
@pytest.mark.vcr
@pytest.mark.asyncio
class TestEdgarDocumentDocumentedExampleVCR:
    async def test_documented_apple_url_example_reads_the_exhibit(self, memoized_find):
        set_identity("Test User test@test.com")

        result = await edgar_document(action="read", url=APPLE_EXHIBIT_URL)

        assert result.success is True, result.error
        assert result.data["document"]["filename"] == "a10-qexhibit32106282025.htm"
        assert result.data["document"]["document_type"] == "EX-32.1"
        assert result.data["source"]["period_of_report"] == "2025-06-28"
        assert result.data["source"]["selected_by"] == "url"
        assert "I, Timothy D\\. Cook, certify" in result.data["text"]
        assert "for the period ended June 28, 2025" in result.data["text"]
