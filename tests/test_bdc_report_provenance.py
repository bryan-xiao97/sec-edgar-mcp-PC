"""BDC report-year provenance, the latest-year cache and honest NOT_A_BDC.

Final-review fix wave for the Private Credit Evidence Access branch:

- **I1 / deferred #12.** Since GH #1146, a bare ``get_bdc_list()`` is the
  union of the last three SEC BDC Reports. ``lookup_bdc`` used it for "the
  latest year", so a row that came from the 2025 report was labelled 2026,
  and ``is_active`` then trusted that row's stale ``last_filing_date`` (ARCC:
  2025-05-29, "inactive" from 2026-11-29). The fakes here model the union at
  the ``fetch_bdc_report`` level, so the real ``get_bdc_list`` and
  ``_combined_bdc_report`` run.
- **I2 / deferred #19.** ``get_latest_bdc_report_year`` was ``lru_cache``d for
  the process lifetime, so one timeout on the current year's probe pinned last
  year forever. Only a confirmed year is cached now, and only for a TTL.
- **I3.** ``NOT_A_BDC`` said "CIK ... is not a Business Development Company"
  for BDCs the SEC report simply omits (Sixth Street Specialty Lending, CIK
  1508655). It now says only what is known: not listed in the years checked.
"""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd
import pytest

from edgar.ai.mcp.tools.bdc import identity as bdc_identity
from edgar.ai.mcp.tools.fund import edgar_fund
from edgar.bdc import reference
from edgar.bdc.reference import get_bdc_list, lookup_bdc
from tests.test_mcp_bdc_portfolio import (
    _CompanyWithFilings,
    _FakeCompany,
    _FakeFiling,
    _make_investment_batch,
    _patch_extraction,
    _patch_selection,
)

ARCC = 1287750
MAIN = 1396440
TSLX = 1508655

ARCC_2025_ROW_FILED = date(2025, 5, 29)


def _report(rows) -> pd.DataFrame:
    """A normalised BDC report frame, as `fetch_bdc_report` returns it."""
    return pd.DataFrame([
        {'file_number': '814-00001', 'cik': cik, 'registrant_name': name,
         'city': 'NEW YORK', 'state': 'NY', 'zip_code': '10019',
         'last_filing_date': pd.Timestamp(filed), 'last_filing_type': '10-K'}
        for cik, name, filed in rows
    ])


def _reports():
    """The measured shape: ARCC is in 2024 and 2025 but not in 2026."""
    return {
        2026: _report([(MAIN, 'MAIN STREET CAPITAL CORP', '2026-05-01'),
                       (9000001, 'NEW BDC', '2026-05-02')]),
        2025: _report([(MAIN, 'MAIN STREET CAPITAL CORP', '2025-05-01'),
                       (ARCC, 'ARES CAPITAL CORP', str(ARCC_2025_ROW_FILED))]),
        2024: _report([(ARCC, 'ARES CAPITAL CORP', '2024-05-30')]),
    }


@pytest.fixture
def per_year_reports(monkeypatch):
    """Fake `fetch_bdc_report` per year (latest 2026). Returns the dict and
    the list of years requested, so a test can change or inspect both."""
    reports = _reports()
    requested = []

    def fake_fetch(year=None):
        requested.append(year)
        if year is None:
            year = 2026
        if year not in reports:
            raise httpx.HTTPStatusError(
                f"404 for {year}", request=httpx.Request("GET", "https://www.sec.gov/x"),
                response=httpx.Response(404),
            )
        return reports[year]

    monkeypatch.setattr(reference, 'fetch_bdc_report', fake_fetch)
    monkeypatch.setattr(reference, 'get_latest_bdc_report_year', lambda: 2026)
    return reports, requested


class _FakeDate(date):
    """`date.today()` frozen at 2026-12-01, after ARCC's 2025 row goes stale."""

    @classmethod
    def today(cls):
        return cls(2026, 12, 1)


# =============================================================================
# I1: lookup_bdc reports the year the matching row actually came from
# =============================================================================


@pytest.mark.fast
class TestLookupReportYear:

    def test_premise_the_union_has_arcc_and_the_latest_report_does_not(self, per_year_reports):
        assert get_bdc_list().get_by_cik(ARCC) is not None
        assert get_bdc_list(2026).get_by_cik(ARCC) is None

    def test_a_row_from_an_older_report_is_labelled_with_that_year(self, per_year_reports):
        bdc = lookup_bdc(cik=ARCC)

        assert bdc.report_year == 2025
        assert bdc.last_filing_date == ARCC_2025_ROW_FILED

    def test_a_row_from_the_latest_report_is_labelled_latest(self, per_year_reports):
        bdc = lookup_bdc(cik=MAIN)

        assert bdc.report_year == 2026
        assert bdc.last_filing_date == date(2026, 5, 1)

    def test_each_year_is_read_exactly_never_through_the_union(self, per_year_reports):
        _, requested = per_year_reports

        lookup_bdc(cik=ARCC)

        assert requested == [2026, 2025]

    def test_a_cik_in_no_report_is_none(self, per_year_reports):
        assert lookup_bdc(cik=TSLX) is None


@pytest.mark.fast
class TestIsActiveAfterTheStaleRowExpires:
    """P1-M5 as intended: on 2026-12-01 ARCC's 2025 row alone says inactive,
    but the response takes `is_active` from ARCC's own latest filing."""

    def test_identity_is_active_comes_from_the_company(self, per_year_reports, monkeypatch):
        monkeypatch.setattr(reference, 'date', _FakeDate)
        company = _CompanyWithFilings(ARCC, latest_filed=date(2026, 11, 3))
        bdc = lookup_bdc(cik=ARCC)

        assert bdc.is_active is False  # the stale 2025 row, read on 2026-12-01
        assert bdc_identity._bdc_is_active(bdc, company) is True
        assert company.get_filings_calls == 1

    @pytest.mark.asyncio
    async def test_bdc_portfolio_response_on_2026_12_01(self, per_year_reports, monkeypatch):
        monkeypatch.setattr(reference, 'date', _FakeDate)
        company = _CompanyWithFilings(ARCC, latest_filed=date(2026, 11, 3))
        monkeypatch.setattr(reference.BDCEntity, 'get_company', lambda self: company)
        _patch_selection(monkeypatch, _FakeFiling())
        _patch_extraction(monkeypatch, _make_investment_batch(2))

        result = await edgar_fund(action="bdc_portfolio", identifier=str(ARCC))

        assert result.success is True, (result.error_code, result.error)
        assert result.data["bdc_report_year"] == 2025
        assert result.data["is_active"] is True


@pytest.mark.fast
class TestNoExtraDownloadForTheExplicitLatestYear:
    """Deferred #12: reading the latest year by its explicit year costs no
    download beyond what the union already made, because `_combined_bdc_report`
    fetches every year under the same `fetch_bdc_report(year)` cache key."""

    CSV_HEADER = "file_number,cik,registrant_name,city,state,zip_code,last_filling_date,last_filling_type\n"

    @pytest.fixture(autouse=True)
    def _fresh_caches(self):
        reference.fetch_bdc_report.cache_clear()
        reference.get_latest_bdc_report_year.cache_clear()
        yield
        reference.fetch_bdc_report.cache_clear()
        reference.get_latest_bdc_report_year.cache_clear()

    def _transport(self, monkeypatch):
        csv = {
            2026: self.CSV_HEADER + f"814-1,{MAIN},MAIN STREET CAPITAL CORP,HOUSTON,TX,77056,05/01/26,10-K\n",
            2025: self.CSV_HEADER + f"814-2,{ARCC},ARES CAPITAL CORP,NEW YORK,NY,10167,05/29/25,10-K\n",
            2024: self.CSV_HEADER + f"814-2,{ARCC},ARES CAPITAL CORP,NEW YORK,NY,10167,05/30/24,10-K\n",
        }
        calls = []

        class _Response:
            def __init__(self, year):
                self.status_code = 200 if year in csv else 404
                self.text = csv.get(year, "")

            def raise_for_status(self):
                if self.status_code != 200:
                    raise httpx.HTTPStatusError(
                        "404", request=httpx.Request("GET", "https://www.sec.gov/x"),
                        response=httpx.Response(404),
                    )

        def _get(url, **kwargs):
            calls.append(url)
            return _Response(int(url.rsplit("-", 1)[1].split(".")[0]))

        monkeypatch.setattr(reference, 'get_with_retry', _get)
        return calls

    def test_lookup_after_the_union_makes_no_request(self, monkeypatch):
        calls = self._transport(monkeypatch)

        assert get_bdc_list().get_by_cik(ARCC) is not None
        before = list(calls)
        bdc = lookup_bdc(cik=ARCC)

        assert calls == before
        assert bdc.report_year == 2025


# =============================================================================
# I2: a transient probe failure is never cached; a confirmed year expires
# =============================================================================


class _Probe:
    """Fake `get_with_retry` for the year probe. `status[year]` is an HTTP
    status or an exception to raise; years not listed answer 404."""

    def __init__(self, status):
        self.status = dict(status)
        self.calls = []

    def __call__(self, url, **kwargs):
        year = int(url.rsplit("-", 1)[1].split(".")[0])
        self.calls.append(year)
        outcome = self.status.get(year, 404)
        if isinstance(outcome, BaseException):
            raise outcome
        return type("R", (), {"status_code": outcome})()


@pytest.mark.fast
class TestLatestReportYearCache:

    @pytest.fixture(autouse=True)
    def _clear(self):
        reference.get_latest_bdc_report_year.cache_clear()
        yield
        reference.get_latest_bdc_report_year.cache_clear()

    def test_a_timeout_on_the_newest_year_is_not_pinned(self, monkeypatch):
        probe = _Probe({2026: httpx.ConnectTimeout("blip"), 2025: 200})
        monkeypatch.setattr(reference, 'get_with_retry', probe)

        assert reference.get_latest_bdc_report_year() == 2025  # best answer during the blip

        probe.status[2026] = 200  # SEC recovers
        assert reference.get_latest_bdc_report_year() == 2026

    def test_a_non_404_answer_on_a_newer_year_is_not_confirmation(self, monkeypatch):
        probe = _Probe({2026: 503, 2025: 200})
        monkeypatch.setattr(reference, 'get_with_retry', probe)

        assert reference.get_latest_bdc_report_year() == 2025

        probe.status[2026] = 200
        assert reference.get_latest_bdc_report_year() == 2026

    def test_the_fallback_year_is_not_cached(self, monkeypatch):
        probe = _Probe({year: httpx.ConnectError("offline") for year in range(2016, 2100)})
        monkeypatch.setattr(reference, 'get_with_retry', probe)

        assert reference.get_latest_bdc_report_year() == reference._BDC_REPORT_FALLBACK_YEAR

        probe.status = {2026: 200}
        assert reference.get_latest_bdc_report_year() == 2026

    def test_a_confirmed_year_is_reused_until_the_ttl_then_reprobed(self, monkeypatch):
        now = [1_000.0]
        monkeypatch.setattr(reference._latest_year_cache, 'clock', lambda: now[0])
        probe = _Probe({2025: 200})  # 2026 not yet published: a definitive 404
        monkeypatch.setattr(reference, 'get_with_retry', probe)

        assert reference.get_latest_bdc_report_year() == 2025
        calls_after_first = len(probe.calls)

        probe.status[2026] = 200  # SEC publishes the 2026 report
        now[0] += reference.LATEST_REPORT_YEAR_TTL_SECONDS - 1
        assert reference.get_latest_bdc_report_year() == 2025  # still within the TTL
        assert len(probe.calls) == calls_after_first  # and no request made

        now[0] += 2  # past the TTL
        assert reference.get_latest_bdc_report_year() == 2026

    def test_the_ttl_is_24_hours(self):
        assert reference.LATEST_REPORT_YEAR_TTL_SECONDS == 24 * 60 * 60


# =============================================================================
# I3: NOT_A_BDC states only what is known
# =============================================================================


def _patch_unlisted_company(monkeypatch, cik=TSLX):
    monkeypatch.setattr("edgar.bdc.reference.lookup_bdc", lambda **kwargs: None)
    monkeypatch.setattr(bdc_identity, "resolve_company", lambda identifier: _FakeCompany(cik=cik))


def _assert_honest_not_listed(result, cik, years_text):
    assert result.success is False
    assert result.error_code == "NOT_A_BDC"
    assert result.error == (
        f"CIK {cik} is not listed in the SEC BDC Report for {years_text}. That report omits "
        f"some BDCs, so this does not establish whether CIK {cik} is a Business Development Company."
    )
    assert "is not a Business Development Company" not in result.error
    assert any("omits some BDCs" in s for s in result.suggestions)


@pytest.mark.fast
@pytest.mark.asyncio
class TestNotABdcWording:

    async def test_cik_path_names_the_years_checked(self, monkeypatch):
        _patch_unlisted_company(monkeypatch)
        monkeypatch.setattr(bdc_identity, "_report_years_checked", lambda: [2024, 2025, 2026])

        result = await edgar_fund(action="bdc_portfolio", identifier=str(TSLX))

        _assert_honest_not_listed(result, TSLX, "2024, 2025 and 2026")

    async def test_accession_path_names_the_years_checked(self, monkeypatch):
        _patch_selection(monkeypatch, _FakeFiling(cik=TSLX, accession_number="0001508655-26-000001"))
        monkeypatch.setattr("edgar.bdc.reference.lookup_bdc", lambda **kwargs: None)
        monkeypatch.setattr(bdc_identity, "_report_years_checked", lambda: [2025, 2026])

        result = await edgar_fund(action="bdc_portfolio", accession_number="0001508655-26-000001")

        _assert_honest_not_listed(result, TSLX, "2025 and 2026")

    async def test_unknown_years_are_not_invented(self, monkeypatch):
        _patch_unlisted_company(monkeypatch)
        monkeypatch.setattr(bdc_identity, "_report_years_checked", lambda: None)

        result = await edgar_fund(action="bdc_nonaccrual", identifier=str(TSLX))

        assert result.error_code == "NOT_A_BDC"
        assert result.error.startswith(f"CIK {TSLX} is not listed in the SEC BDC Report years checked.")
        assert "is not a Business Development Company" not in result.error


@pytest.mark.fast
class TestReportYearsChecked:
    """`_report_years_checked` names only the years whose report was read."""

    def test_all_three_years_answered(self, per_year_reports):
        assert bdc_identity._report_years_checked() == [2024, 2025, 2026]

    def test_a_year_that_failed_is_not_named(self, per_year_reports):
        reports, _ = per_year_reports
        del reports[2024]

        assert bdc_identity._report_years_checked() == [2025, 2026]

    def test_years_text(self):
        assert bdc_identity._years_text([2026]) == "2026"
        assert bdc_identity._years_text([2025, 2026]) == "2025 and 2026"
        assert bdc_identity._years_text([2024, 2025, 2026]) == "2024, 2025 and 2026"
