"""
Tests for edgar_notes and edgar_read's chosen-filing selection and continuation
(Task 6).

Fast tests monkeypatch filing selection (`resolve_report_filing`, patched at
its source module `edgar.ai.mcp.tools.selection` -- both tools import it
lazily inside their filing-selection helper, mirroring `fund.py`'s
convention, so tests can substitute a fake filing without touching the
network) and, for edgar_read, `_extract_section` (the one call that would
otherwise need a real filing object). They verify table-row paging, note
context text paging, and edgar_read's cursor/section argument checks and
fallback flag without any of that.

Network tests pin behaviour against real filings (constraints rule 7a):
- ARCC's 10-Q (CIK 1287750, accession 0001628280-26-050307) runs LIVE, no
  VCR cassette -- any test that parses its XBRL/notes touches the ~88 MB full
  submission, an oversized commit (see the Fixture rule in the task brief).
- Princeton Capital Corp's 10-Q (CIK 845385, accession 0001213900-26-090000)
  is the small-BDC deterministic VCR fixture from Task 3 (see
  tests/test_bdc_filing_scoped.py for how it was chosen); one VCR test each
  for edgar_notes and edgar_read pins the same paging behaviour against a
  recorded cassette.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import pytest

from edgar.ai.mcp.tools import continuation, reader as reader_module
from edgar.ai.mcp.tools.continuation import ResultCache
from edgar.ai.mcp.tools.notes import edgar_notes
from edgar.ai.mcp.tools.reader import edgar_read
from edgar.ai.mcp.tools.selection import FilingSelection


# =============================================================================
# Fakes -- edgar_notes
# =============================================================================

class _FakeTable:
    """Just enough of a Statement for _build_table_info's happy path."""

    def __init__(self, role: str, df: pd.DataFrame):
        self.role_or_type = role
        self._df = df

    def render(self):
        return None

    def to_dataframe(self):
        return self._df


class _FakeNote:
    """Just enough of a Note for edgar_notes' matched-note path."""

    def __init__(
        self,
        number: int,
        title: str,
        short_name: Optional[str] = None,
        tables: Optional[list] = None,
        context_text: str = "",
        expands: Optional[list] = None,
        expands_statements: Optional[list] = None,
    ):
        self.number = number
        self.title = title
        self.short_name = short_name or title.lower()
        self.tables = tables or []
        self.policies: list = []
        self.details: list = []
        self.expands = expands or []
        self.expands_statements = expands_statements or []
        self._context_text = context_text

    @property
    def table_count(self) -> int:
        return len(self.tables)

    def to_context(self, detail: str = "standard") -> str:
        return self._context_text


class _FakeNotes:
    """Just enough of Notes for edgar_notes: iteration, by-number lookup, search."""

    def __init__(self, notes: list[_FakeNote]):
        self._notes = notes
        self._by_number = {n.number: n for n in notes}

    def __iter__(self):
        return iter(self._notes)

    def __len__(self):
        return len(self._notes)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._by_number.get(key)
        return None

    def search(self, keyword: str) -> list[_FakeNote]:
        needle = keyword.lower()
        return [n for n in self._notes if needle in n.title.lower()]


class _FakeReportObj:
    """Just enough of a TenK/TenQ report object for edgar_notes."""

    def __init__(self, notes: _FakeNotes, period_of_report: str = "2026-06-30"):
        self.notes = notes
        self.period_of_report = period_of_report


class _FakeNotesFiling:
    """Just enough of a Filing for edgar_notes' happy path."""

    def __init__(
        self,
        obj: _FakeReportObj,
        accession_number: str = "0001213900-26-090000",
        form: str = "10-Q",
    ):
        self._obj = obj
        self.accession_number = accession_number
        self.form = form
        self.cik = 845385
        self.company = "Princeton Capital Corp"
        self.filing_date = "2026-08-01"
        self.homepage_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.report_date = "2026-06-30"
        self.obj_call_count = 0

    def obj(self):
        self.obj_call_count += 1
        return self._obj


def _patch_notes_selection(monkeypatch, filing):
    monkeypatch.setattr(
        "edgar.ai.mcp.tools.selection.resolve_report_filing",
        lambda **kwargs: FilingSelection(filing=filing, selected_by="latest"),
    )


def _patch_selection_by_accession(monkeypatch, filings, latest=None, valid_identifier="PFX"):
    """A resolver fake that honours the accession path the way the real one
    does (constraints rule 7b / P1-H2): selects by `accession_number` when
    given, raising SELECTION_MISMATCH for any OTHER identifier (simulating
    the real CIK cross-check); by exact `period`; else `latest`. Records
    every call's kwargs. Shared by the notes and read cursor-only tests
    below; mirrors `_patch_selection_by_accession` in
    tests/test_mcp_bdc_portfolio.py."""
    from edgar.ai.mcp.tools.selection import FilingSelectionError

    by_acc = {f.accession_number: f for f in filings}
    calls = []

    def _resolve(**kwargs):
        calls.append(kwargs)
        acc = kwargs.get("accession_number")
        if acc:
            filing = by_acc.get(acc)
            if filing is None:
                raise FilingSelectionError("not found", error_code="FILING_NOT_FOUND")
            identifier = kwargs.get("identifier")
            if identifier is not None and identifier != valid_identifier:
                raise FilingSelectionError("mismatch", error_code="SELECTION_MISMATCH")
            return FilingSelection(filing=filing, selected_by="accession")
        if kwargs.get("period"):
            match = next((f for f in filings if f.report_date == kwargs["period"]), None)
            if match is None:
                raise FilingSelectionError("not found", error_code="PERIOD_NOT_FOUND")
            return FilingSelection(filing=match, selected_by="period")
        return FilingSelection(filing=latest or filings[0], selected_by="latest")

    monkeypatch.setattr("edgar.ai.mcp.tools.selection.resolve_report_filing", _resolve)
    return calls


def _holds_xbrl(value, _seen=None) -> bool:
    """True if `value` -- or anything reachable by walking it (dict values,
    or any other iterable's elements, e.g. a Notes container) -- has an
    `_xbrl` attribute (P2-M4: that is exactly what pins the parsed XBRL
    model, e.g. `edgar.xbrl.notes.Note._xbrl`)."""
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return False
    _seen.add(id(value))
    if hasattr(value, "_xbrl"):
        return True
    if isinstance(value, dict):
        return any(_holds_xbrl(v, _seen) for v in value.values())
    if isinstance(value, (str, bytes)) or value is None:
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False
    return any(_holds_xbrl(v, _seen) for v in iterator)


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch):
    # text_cache is isolated too (not just results_cache): several notes
    # tests below reuse the SAME default accession number across different
    # fake filings, and notes.py now stores per-table/context evidence in
    # the shared, module-level text_cache (P2-M4) -- without this, one
    # test's cached rows could leak into another test that decodes a
    # cursor for the "same" accession/note/table but different content.
    monkeypatch.setattr(continuation, "results_cache", ResultCache(max_entries=8))
    monkeypatch.setattr(continuation, "text_cache", ResultCache(max_entries=64, max_bytes=32 * 1024 * 1024))


# =============================================================================
# edgar_notes: table row paging
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestNotesTablePaging:
    async def test_continuation_reuses_parsed_notes_across_fresh_filings(self, monkeypatch):
        table = _FakeTable("Schedule of Debt", pd.DataFrame({"principal": [1, 2, 3, 4]}))
        note = _FakeNote(1, "Debt", tables=[table])
        first_filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        second_filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        second_filing.company = "Current selection entity"
        selections = iter([
            FilingSelection(filing=first_filing, selected_by="latest"),
            FilingSelection(filing=second_filing, selected_by="accession"),
        ])
        monkeypatch.setattr(
            "edgar.ai.mcp.tools.selection.resolve_report_filing", lambda **kwargs: next(selections)
        )

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=2)
        assert first.success is True
        table_info = first.data["notes"][0]["tables"][0]
        assert table_info["data"] == [{"principal": 1}, {"principal": 2}]
        cursor = table_info["next_cursor"]

        second = await edgar_notes(
            accession_number=first_filing.accession_number, topic="debt", cursor=cursor, limit=2
        )
        assert second.success is True
        assert second.data["table"] == [{"principal": 3}, {"principal": 4}]
        assert second.data["page"]["next_cursor"] is None
        assert second.data["source"]["entity"] == "Current selection entity"
        assert second.data["source"]["selected_by"] == "accession"
        assert first_filing.obj_call_count == 1
        assert second_filing.obj_call_count == 0

    async def test_walks_all_rows_exactly_once(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2, 3, 4, 5]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", "debt", tables=[table], context_text="short")
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=2)
        assert first.success is True

        table_info = first.data["notes"][0]["tables"][0]
        assert table_info["rows"] == 5
        assert table_info["rows_total"] == 5
        assert table_info["columns"] == ["principal"]

        collected = [row["principal"] for row in table_info["data"]]
        cursor = table_info["next_cursor"]
        assert cursor is not None

        seen_cursors = set()
        while cursor:
            assert cursor not in seen_cursors, "cursor repeated -- would loop forever"
            seen_cursors.add(cursor)
            page = await edgar_notes(identifier="PFX", topic="debt", cursor=cursor, limit=2)
            assert page.success is True, page.error
            assert page.data["note"] == {"number": 1, "title": "Debt"}
            collected.extend(row["principal"] for row in page.data["table"])
            cursor = page.data["page"]["next_cursor"]

        assert collected == [1, 2, 3, 4, 5]

    async def test_next_cursor_null_when_everything_fits_on_one_page(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        result = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=10)
        table_info = result.data["notes"][0]["tables"][0]
        assert "next_cursor" not in table_info

    async def test_cursor_referencing_a_note_no_longer_present_is_cursor_mismatch(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2, 3]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=1)
        cursor = first.data["notes"][0]["tables"][0]["next_cursor"]
        assert cursor is not None

        # Note 1 is gone from this "later" filing -- same accession, only note 2 remains.
        other_note = _FakeNote(2, "Revenue")
        filing2 = _FakeNotesFiling(_FakeReportObj(_FakeNotes([other_note])), accession_number=filing.accession_number)
        _patch_notes_selection(monkeypatch, filing2)
        # Re-extract after eviction so this test still covers a missing note.
        # (P2-M4: notes.py caches per-table/context evidence in text_cache,
        # not the whole Notes/report object in results_cache -- clearing
        # results_cache alone would not force a reparse here.)
        monkeypatch.setattr(continuation, "text_cache", ResultCache(max_entries=64, max_bytes=32 * 1024 * 1024))

        result = await edgar_notes(identifier="PFX", topic="debt", cursor=cursor)
        assert result.success is False
        assert result.error_code == "CURSOR_MISMATCH"

    async def test_cursor_minted_under_one_topic_rejected_when_replayed_under_another(self, monkeypatch):
        """A table cursor is validated against the CURRENT call's topic, not
        the topic baked into the cursor itself -- so a client that changes
        topic between calls gets CURSOR_MISMATCH, not the other topic's next
        page."""
        df = pd.DataFrame({"principal": [1, 2, 3, 4, 5]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=2)
        cursor = first.data["notes"][0]["tables"][0]["next_cursor"]
        assert cursor is not None

        wrong_topic = await edgar_notes(identifier="PFX", topic="revenue", cursor=cursor, limit=2)
        assert wrong_topic.success is False
        assert wrong_topic.error_code == "CURSOR_MISMATCH"

        # The same cursor, replayed with the topic it was actually minted
        # under, still continues correctly.
        right_topic = await edgar_notes(identifier="PFX", topic="debt", cursor=cursor, limit=2)
        assert right_topic.success is True, right_topic.error
        assert [row["principal"] for row in right_topic.data["table"]] == [3, 4]


# =============================================================================
# edgar_notes: context text paging
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestNotesContextPaging:
    async def test_context_paging_reassembles_full_text(self, monkeypatch):
        big_text = ("A" * 6500) + "\n\n" + ("B" * 3500)
        note = _FakeNote(2, "Revenue Recognition", context_text=big_text)
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        first = await edgar_notes(identifier="PFX", topic="revenue", detail="standard")
        assert first.success is True

        note_data = first.data["notes"][0]
        context_page = note_data["context_page"]
        assert context_page["offset"] == 0
        assert context_page["total_chars"] == len(big_text)
        cursor = context_page["next_cursor"]
        assert cursor is not None

        collected = note_data["context"]
        seen_cursors = set()
        while cursor:
            assert cursor not in seen_cursors
            seen_cursors.add(cursor)
            page = await edgar_notes(identifier="PFX", topic="revenue", detail="standard", cursor=cursor)
            assert page.success is True, page.error
            collected += page.data["context"]
            cursor = page.data["page"]["next_cursor"]

        assert collected == big_text

    async def test_short_context_has_no_next_cursor(self, monkeypatch):
        note = _FakeNote(2, "Revenue Recognition", context_text="short text")
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        result = await edgar_notes(identifier="PFX", topic="revenue")
        note_data = result.data["notes"][0]
        assert note_data["context"] == "short text"
        assert note_data["context_page"]["next_cursor"] is None

    async def test_cursor_minted_under_one_detail_rejected_when_replayed_under_another(self, monkeypatch):
        """A context cursor is validated against the CURRENT call's detail
        level, not the detail baked into the cursor itself -- so a client
        that changes detail between calls gets CURSOR_MISMATCH, not the
        other detail level's next page."""
        big_text = ("A" * 6500) + "\n\n" + ("B" * 3500)
        note = _FakeNote(2, "Revenue Recognition", context_text=big_text)
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        first = await edgar_notes(identifier="PFX", topic="revenue", detail="standard")
        cursor = first.data["notes"][0]["context_page"]["next_cursor"]
        assert cursor is not None

        wrong_detail = await edgar_notes(identifier="PFX", topic="revenue", detail="full", cursor=cursor)
        assert wrong_detail.success is False
        assert wrong_detail.error_code == "CURSOR_MISMATCH"

        # The same cursor, replayed with the detail it was actually minted
        # under, still continues correctly.
        right_detail = await edgar_notes(identifier="PFX", topic="revenue", detail="standard", cursor=cursor)
        assert right_detail.success is True, right_detail.error
        assert first.data["notes"][0]["context"] + right_detail.data["context"] == big_text


# =============================================================================
# edgar_notes: source block and argument validation
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestNotesArgumentsAndSource:
    async def test_missing_identifier_and_accession_is_invalid_arguments(self):
        result = await edgar_notes()
        assert result.success is False
        assert result.error_code == "INVALID_ARGUMENTS"

    async def test_response_carries_source_block(self, monkeypatch):
        note = _FakeNote(1, "Debt")
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        result = await edgar_notes(identifier="PFX", topic="debt")
        assert result.success is True
        source = result.data["source"]
        assert source["accession_number"] == "0001213900-26-090000"
        assert source["selected_by"] == "latest"

    async def test_garbage_cursor_is_invalid_cursor(self, monkeypatch):
        """Silence check: a cursor that isn't decodable at all fails
        peek_cursor before any note lookup, as INVALID_CURSOR -- not a
        crash, not a silent None."""
        note = _FakeNote(1, "Debt")
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        result = await edgar_notes(identifier="PFX", topic="debt", cursor="not-a-real-cursor!!")
        assert result.success is False
        assert result.error_code == "INVALID_CURSOR"


# =============================================================================
# edgar_notes: cursor-only continuation (P1-H2 / constraints rule 7b)
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestNotesCursorOnlyContinuation:
    """A cursor alone is enough; a supplied topic/detail/accession_number/
    identifier must agree with the cursor."""

    async def test_cursor_alone_walks_all_table_rows(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2, 3, 4, 5]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_selection_by_accession(monkeypatch, [filing])

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=2)
        assert first.success is True
        collected = [row["principal"] for row in first.data["notes"][0]["tables"][0]["data"]]
        cursor = first.data["notes"][0]["tables"][0]["next_cursor"]
        assert cursor is not None

        pages = 0
        while cursor:
            page = await edgar_notes(cursor=cursor)
            assert page.success is True, page.error
            assert page.data["note"] == {"number": 1, "title": "Debt"}
            collected.extend(row["principal"] for row in page.data["table"])
            cursor = page.data["page"]["next_cursor"]
            pages += 1
            assert pages < 10

        assert collected == [1, 2, 3, 4, 5]

    async def test_cursor_alone_walks_context_text(self, monkeypatch):
        big_text = ("A" * 6500) + "\n\n" + ("B" * 3500)
        note = _FakeNote(2, "Revenue Recognition", context_text=big_text)
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_selection_by_accession(monkeypatch, [filing])

        first = await edgar_notes(identifier="PFX", topic="revenue", detail="standard")
        assert first.success is True
        cursor = first.data["notes"][0]["context_page"]["next_cursor"]
        assert cursor is not None
        collected = first.data["notes"][0]["context"]

        page = await edgar_notes(cursor=cursor)
        assert page.success is True, page.error
        collected += page.data["context"]
        assert page.data["page"]["next_cursor"] is None
        assert collected == big_text

    async def test_supplied_topic_matching_cursor_is_accepted(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2, 3]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_selection_by_accession(monkeypatch, [filing])

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=1)
        cursor = first.data["notes"][0]["tables"][0]["next_cursor"]

        page = await edgar_notes(topic="debt", cursor=cursor)
        assert page.success is True, page.error

    async def test_supplied_topic_differing_from_cursor_is_cursor_mismatch(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2, 3]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_selection_by_accession(monkeypatch, [filing])

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=1)
        cursor = first.data["notes"][0]["tables"][0]["next_cursor"]

        page = await edgar_notes(topic="revenue", cursor=cursor)
        assert page.success is False
        assert page.error_code == "CURSOR_MISMATCH"

    async def test_supplied_detail_differing_from_cursor_context_is_cursor_mismatch(self, monkeypatch):
        big_text = ("A" * 6500) + "\n\n" + ("B" * 3500)
        note = _FakeNote(2, "Revenue Recognition", context_text=big_text)
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_selection_by_accession(monkeypatch, [filing])

        first = await edgar_notes(identifier="PFX", topic="revenue", detail="standard")
        cursor = first.data["notes"][0]["context_page"]["next_cursor"]

        page = await edgar_notes(detail="full", cursor=cursor)
        assert page.success is False
        assert page.error_code == "CURSOR_MISMATCH"

    async def test_identifier_for_a_different_company_is_selection_mismatch(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2, 3]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_selection_by_accession(monkeypatch, [filing])

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=1)
        cursor = first.data["notes"][0]["tables"][0]["next_cursor"]

        page = await edgar_notes(identifier="OTHER", cursor=cursor)
        assert page.success is False
        assert page.error_code == "SELECTION_MISMATCH"

    async def test_accession_number_differing_from_cursor_is_cursor_mismatch(self, monkeypatch):
        df = pd.DataFrame({"principal": [1, 2, 3]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table])
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        calls = _patch_selection_by_accession(monkeypatch, [filing])

        first = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=1)
        cursor = first.data["notes"][0]["tables"][0]["next_cursor"]
        calls.clear()

        page = await edgar_notes(accession_number="0000000000-26-000001", cursor=cursor)
        assert page.success is False
        assert page.error_code == "CURSOR_MISMATCH"
        assert calls == []  # rejected before any filing selection

    async def test_garbage_cursor_alone_is_invalid_cursor(self, monkeypatch):
        """Silence check: a cursor-only call with a garbage cursor names the
        cursor as the problem, not a missing identifier."""
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([_FakeNote(1, "Debt")])))
        calls = _patch_selection_by_accession(monkeypatch, [filing])

        result = await edgar_notes(cursor="!!!not-a-real-cursor!!!")
        assert result.success is False
        assert result.error_code == "INVALID_CURSOR"
        assert calls == []


# =============================================================================
# edgar_notes: bounded cache never retains parsed XBRL (P2-M4)
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestNotesCacheNeverRetainsXbrl:
    async def test_cached_entries_hold_no_object_with_xbrl_attribute(self, monkeypatch):
        """The notes cache must never retain an object with an `_xbrl`
        attribute -- that is exactly what pinned the whole parsed XBRL
        model (~6.4 MB for Princeton) in the shared, unbounded
        results_cache before this fix."""
        df = pd.DataFrame({"principal": [1, 2, 3]})
        table = _FakeTable("Schedule of Debt", df)
        note = _FakeNote(1, "Debt", tables=[table], context_text="A" * 7000)
        note._xbrl = object()  # simulates edgar.xbrl.notes.Note._xbrl
        filing = _FakeNotesFiling(_FakeReportObj(_FakeNotes([note])))
        _patch_notes_selection(monkeypatch, filing)

        result = await edgar_notes(identifier="PFX", topic="debt", detail="full", limit=1)
        assert result.success is True

        for cache in (continuation.results_cache, continuation.text_cache):
            for key in list(cache._data.keys()):
                value = cache.get(key)
                assert not _holds_xbrl(value), f"cache entry {key!r} retains an object with _xbrl"


# =============================================================================
# Fakes -- edgar_read
# =============================================================================

class _FakeReadFiling:
    """Just enough of a Filing for edgar_read's happy path."""

    def __init__(self, accession_number: str = "0001213900-26-090000", form: str = "10-Q"):
        self.accession_number = accession_number
        self.form = form
        self.cik = 845385
        self.company = "Princeton Capital Corp"
        self.filing_date = "2026-08-01"
        self.homepage_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.report_date = "2026-06-30"

    def obj(self):
        return object()

    def text(self):
        return "raw filing text " * 100


def _patch_read_selection(monkeypatch, filing):
    monkeypatch.setattr(
        "edgar.ai.mcp.tools.selection.resolve_report_filing",
        lambda **kwargs: FilingSelection(filing=filing, selected_by="latest"),
    )


# =============================================================================
# edgar_read: section text paging
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestReadSectionPaging:
    async def test_section_paging_reassembles_full_text(self, monkeypatch):
        text = ("P" * 6500) + "\n\n" + ("Q" * 3000)
        filing = _FakeReadFiling(accession_number="0001111111-26-000001")
        _patch_read_selection(monkeypatch, filing)
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: text)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        assert first.success is True

        page_block = first.data["section_pages"]["mda"]
        assert page_block["total_chars"] == len(text)
        assert page_block["offset"] == 0
        collected = first.data["sections"]["mda"]
        cursor = page_block["next_cursor"]
        assert cursor is not None

        seen_cursors = set()
        while cursor:
            assert cursor not in seen_cursors
            seen_cursors.add(cursor)
            page = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"], cursor=cursor)
            assert page.success is True, page.error
            collected += page.data["sections"]["mda"]
            cursor = page.data["section_pages"]["mda"]["next_cursor"]

        assert collected == text

    async def test_short_section_has_no_next_cursor(self, monkeypatch):
        filing = _FakeReadFiling(accession_number="0001111111-26-000002")
        _patch_read_selection(monkeypatch, filing)
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: "short section text")

        result = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        assert result.data["sections"]["mda"] == "short section text"
        assert result.data["section_pages"]["mda"]["next_cursor"] is None


# =============================================================================
# edgar_read: cursor argument checks
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestReadCursorArguments:
    async def test_cursor_for_wrong_section_is_cursor_mismatch(self, monkeypatch):
        text = "X" * 7000
        filing = _FakeReadFiling(accession_number="0001111111-26-000003")
        _patch_read_selection(monkeypatch, filing)
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: text)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        cursor = first.data["section_pages"]["mda"]["next_cursor"]
        assert cursor is not None

        wrong = await edgar_read(identifier="PFX", form="10-Q", sections=["risk_factors"], cursor=cursor)
        assert wrong.success is False
        assert wrong.error_code == "CURSOR_MISMATCH"

    async def test_cursor_with_multiple_sections_is_invalid_arguments(self):
        result = await edgar_read(
            identifier="PFX", form="10-Q", sections=["mda", "risk_factors"], cursor="anything"
        )
        assert result.success is False
        assert result.error_code == "INVALID_ARGUMENTS"

    async def test_cursor_with_summary_is_invalid_arguments(self):
        result = await edgar_read(identifier="PFX", form="10-Q", sections=["summary"], cursor="anything")
        assert result.success is False
        assert result.error_code == "INVALID_ARGUMENTS"

    async def test_stale_cursor_after_content_changes_is_cursor_stale(self, monkeypatch):
        filing = _FakeReadFiling(accession_number="0001111111-26-000004")
        _patch_read_selection(monkeypatch, filing)
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: "X" * 7000)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        cursor = first.data["section_pages"]["mda"]["next_cursor"]
        assert cursor is not None

        # Same section, different underlying text -- and a fresh filing (new
        # accession-scoped cache key) so the change isn't masked by the cache.
        filing2 = _FakeReadFiling(accession_number=filing.accession_number)
        _patch_read_selection(monkeypatch, filing2)
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: "Y" * 7000)

        # Force a re-extraction past the cache so the changed text is seen.
        import edgar.ai.mcp.tools.continuation as continuation

        monkeypatch.setattr(continuation, "text_cache", continuation.ResultCache(max_entries=8))

        result = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"], cursor=cursor)
        assert result.success is False
        assert result.error_code == "CURSOR_STALE"


# =============================================================================
# edgar_read: cursor-only continuation (P1-H2 / constraints rule 7b)
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestReadCursorOnlyContinuation:
    """A cursor alone is enough; a supplied sections/accession_number/
    identifier must agree with the cursor."""

    async def test_cursor_alone_continues_the_chosen_section(self, monkeypatch):
        text = ("P" * 6500) + "\n\n" + ("Q" * 3000)
        filing = _FakeReadFiling(accession_number="0002222222-26-000001")
        _patch_selection_by_accession(monkeypatch, [filing])
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: text)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        assert first.success is True
        cursor = first.data["section_pages"]["mda"]["next_cursor"]
        assert cursor is not None
        collected = first.data["sections"]["mda"]

        page = await edgar_read(cursor=cursor)
        assert page.success is True, page.error
        assert page.data["source"]["accession_number"] == filing.accession_number
        assert set(page.data["sections"].keys()) == {"mda"}
        collected += page.data["sections"]["mda"]
        assert page.data["section_pages"]["mda"]["next_cursor"] is None
        assert collected == text

    async def test_supplied_matching_section_is_accepted(self, monkeypatch):
        text = "X" * 7000
        filing = _FakeReadFiling(accession_number="0002222222-26-000002")
        _patch_selection_by_accession(monkeypatch, [filing])
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: text)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        cursor = first.data["section_pages"]["mda"]["next_cursor"]

        page = await edgar_read(sections=["mda"], cursor=cursor)
        assert page.success is True, page.error

    async def test_a_different_section_with_the_cursor_is_cursor_mismatch(self, monkeypatch):
        text = "X" * 7000
        filing = _FakeReadFiling(accession_number="0002222222-26-000003")
        _patch_selection_by_accession(monkeypatch, [filing])
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: text)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        cursor = first.data["section_pages"]["mda"]["next_cursor"]

        page = await edgar_read(sections=["risk_factors"], cursor=cursor)
        assert page.success is False
        assert page.error_code == "CURSOR_MISMATCH"

    async def test_identifier_for_a_different_company_is_selection_mismatch(self, monkeypatch):
        text = "X" * 7000
        filing = _FakeReadFiling(accession_number="0002222222-26-000004")
        _patch_selection_by_accession(monkeypatch, [filing])
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: text)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        cursor = first.data["section_pages"]["mda"]["next_cursor"]

        page = await edgar_read(identifier="OTHER", cursor=cursor)
        assert page.success is False
        assert page.error_code == "SELECTION_MISMATCH"

    async def test_accession_number_differing_from_cursor_is_cursor_mismatch(self, monkeypatch):
        text = "X" * 7000
        filing = _FakeReadFiling(accession_number="0002222222-26-000005")
        calls = _patch_selection_by_accession(monkeypatch, [filing])
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: text)

        first = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        cursor = first.data["section_pages"]["mda"]["next_cursor"]
        calls.clear()

        page = await edgar_read(accession_number="0000000000-26-000001", cursor=cursor)
        assert page.success is False
        assert page.error_code == "CURSOR_MISMATCH"
        assert calls == []  # rejected before any filing selection

    async def test_another_tools_cursor_is_rejected_before_any_selection(self, monkeypatch):
        calls = _patch_selection_by_accession(monkeypatch, [_FakeReadFiling(accession_number="0002222222-26-000006")])
        foreign_cursor = continuation.encode_cursor(
            tool="edgar_notes:table", accession="0002222222-26-000006", offset=20, fp="abc", query=None,
        )

        result = await edgar_read(cursor=foreign_cursor)

        assert result.success is False
        assert result.error_code == "CURSOR_MISMATCH"
        assert calls == []

    async def test_garbage_cursor_alone_is_invalid_cursor(self, monkeypatch):
        """Silence check: a cursor-only call with a garbage cursor names the
        cursor as the problem, not a missing identifier/form."""
        calls = _patch_selection_by_accession(monkeypatch, [_FakeReadFiling()])

        result = await edgar_read(cursor="!!!not-a-real-cursor!!!")
        assert result.success is False
        assert result.error_code == "INVALID_CURSOR"
        assert calls == []


# =============================================================================
# edgar_read: raw_text_preview fallback flagged
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestReadFallback:
    async def test_fallback_used_and_reason_are_set(self, monkeypatch):
        filing = _FakeReadFiling(accession_number="0001111111-26-000005")
        _patch_read_selection(monkeypatch, filing)

        def _boom(obj, filing_, section):
            raise ValueError("boom-extract")

        monkeypatch.setattr(reader_module, "_get_section_text_cached", _boom)

        result = await edgar_read(identifier="PFX", form="10-Q", sections=["mda"])
        assert result.success is True
        extracted = result.data["sections"]
        assert extracted["fallback_used"] is True
        assert extracted["fallback_reason"] == "boom-extract"
        assert "raw filing text" in extracted["raw_text_preview"]
        assert extracted["error"] == "boom-extract"


# =============================================================================
# edgar_read: form-required argument validation
# =============================================================================

@pytest.mark.fast
@pytest.mark.asyncio
class TestReadFormRequired:
    async def test_identifier_without_form_is_invalid_arguments(self):
        result = await edgar_read(identifier="PFX")
        assert result.success is False
        assert result.error_code == "INVALID_ARGUMENTS"

    async def test_identifier_and_period_without_form_is_invalid_arguments(self):
        result = await edgar_read(identifier="PFX", period="2026-06-30")
        assert result.success is False
        assert result.error_code == "INVALID_ARGUMENTS"

    async def test_no_params_is_invalid_arguments(self):
        result = await edgar_read()
        assert result.success is False
        assert result.error_code == "INVALID_ARGUMENTS"


# =============================================================================
# Network: ARCC (live, no cassette -- Fixture rule)
# =============================================================================

ARCC_10Q_ACCESSION = "0001628280-26-050307"
ARCC_10Q_PERIOD = "2026-06-30"


@pytest.mark.network
@pytest.mark.asyncio
class TestEdgarNotesARCCLive:
    """Runs LIVE (no VCR) -- parses the ARCC 10-Q's XBRL/FilingSummary, whose
    full submission is ~88 MB (constraints rule 7a)."""

    async def test_period_selects_ground_truth_filing(self):
        from edgar import set_identity

        set_identity("Test User test@test.com")
        result = await edgar_notes(
            identifier="ARCC", form="10-Q", period=ARCC_10Q_PERIOD, topic="debt"
        )
        assert result.success is True, result.error
        assert result.data["source"]["accession_number"] == ARCC_10Q_ACCESSION
        assert result.data["source"]["selected_by"] == "period"


@pytest.mark.network
@pytest.mark.asyncio
class TestEdgarReadARCCLive:
    """Runs LIVE (no VCR) -- parses the ARCC 10-Q's structured object to read
    its MD&A section (constraints rule 7a)."""

    async def test_mda_section_pages_reassemble_full_text(self):
        from edgar import set_identity

        set_identity("Test User test@test.com")
        first = await edgar_read(
            identifier="ARCC", form="10-Q", period=ARCC_10Q_PERIOD, sections=["mda"]
        )
        assert first.success is True, first.error
        assert first.data["source"]["accession_number"] == ARCC_10Q_ACCESSION

        page_block = first.data["section_pages"]["mda"]
        total_chars = page_block["total_chars"]
        assert total_chars > 6000
        assert len(first.data["sections"]["mda"]) <= 6000

        collected = first.data["sections"]["mda"]
        cursor = page_block["next_cursor"]
        assert cursor is not None

        seen_cursors = set()
        while cursor:
            assert cursor not in seen_cursors
            seen_cursors.add(cursor)
            page = await edgar_read(
                identifier="ARCC", form="10-Q", period=ARCC_10Q_PERIOD, sections=["mda"], cursor=cursor
            )
            assert page.success is True, page.error
            collected += page.data["sections"]["mda"]
            cursor = page.data["section_pages"]["mda"]["next_cursor"]

        assert len(collected) == total_chars

    async def test_walking_all_pages_with_the_cursor_alone(self):
        """P1-H2 / rule 7b, against the real ~88 MB submission: after page
        1, every later call sends only `cursor` -- no identifier, form,
        period, or sections -- and still reassembles the exact same total
        length the repeat-args walk above gets."""
        from edgar import set_identity

        set_identity("Test User test@test.com")
        first = await edgar_read(
            identifier="ARCC", form="10-Q", period=ARCC_10Q_PERIOD, sections=["mda"]
        )
        assert first.success is True, first.error
        assert first.data["source"]["accession_number"] == ARCC_10Q_ACCESSION

        page_block = first.data["section_pages"]["mda"]
        total_chars = page_block["total_chars"]
        collected = first.data["sections"]["mda"]
        cursor = page_block["next_cursor"]
        assert cursor is not None

        seen_cursors = set()
        while cursor:
            assert cursor not in seen_cursors
            seen_cursors.add(cursor)
            page = await edgar_read(cursor=cursor)
            assert page.success is True, page.error
            assert page.data["source"]["accession_number"] == ARCC_10Q_ACCESSION
            collected += page.data["sections"]["mda"]
            cursor = page.data["section_pages"]["mda"]["next_cursor"]

        assert len(collected) == total_chars


# =============================================================================
# Network: Princeton Capital Corp (deterministic VCR fixture, Task 3)
# =============================================================================

PRINCETON_10Q_ACCESSION = "0001213900-26-090000"
PRINCETON_10Q_PERIOD = "2026-06-30"


@pytest.mark.network
@pytest.mark.vcr
@pytest.mark.asyncio
class TestEdgarNotesPrincetonVCR:
    """Deterministic small-BDC ground truth (rule 7a): Princeton Capital's
    2026-06-30 10-Q has no 'debt' note (it is an investment company, not a
    corporate borrower) but does have a 'Commitments and Contingencies' note
    (number 8), used here for the same accession-selection/source assertion
    the ARCC live test makes."""

    async def test_accession_selects_princeton_filing(self):
        from edgar import set_identity

        set_identity("Test User test@test.com")
        result = await edgar_notes(accession_number=PRINCETON_10Q_ACCESSION, topic="commitments")
        assert result.success is True, result.error
        assert result.data["source"]["accession_number"] == PRINCETON_10Q_ACCESSION
        assert result.data["source"]["selected_by"] == "accession"
        assert result.data["notes"][0]["number"] == 8



# NOTE: an equivalent edgar_read Princeton VCR test (MD&A section paging) was
# tried and dropped -- its recorded cassette measured 11.7 MB, over the
# 10 MB cap (constraints rule 7a). edgar_read's section extraction reaches
# for filing content beyond what edgar_notes' XBRL-only path touches, so it
# does not fit the same small-BDC fixture within the cap. The ARCC live test
# above (TestEdgarReadARCCLive) covers this behavior instead.
