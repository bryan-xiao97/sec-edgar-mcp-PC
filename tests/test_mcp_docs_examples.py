"""Verify documented MCP tool-call examples against the tools' registered
schemas, and that the documented cursor-only continuation flows actually run.

Scope (constraints rule 7c: ``docs/ai/`` was deleted; user-facing docs for
this project live ONLY in the sources below):
  - ``edgar/ai/mcp/docs/MCP_QUICKSTART.md``
  - every registered tool's own ``description=`` text (marked JSON examples
    AND the "Examples:" prose bullets, for the tools that use that style)
  - ``SERVER_INSTRUCTIONS`` (``edgar/ai/mcp/server.py``)
  - ``edgar/ai/mcp/tools/prompts.py`` (the rendered prompt text)
  - the skill YAMLs under ``edgar/ai/skills/`` that mention an MCP tool call

Two independent things are checked:
  1. Every documented example uses parameter names and enum values the
     tool's registered schema actually accepts (``_check_source_examples``),
     including a filing selector or a cursor.
  2. A cursor alone is enough to continue: fast tests drive the real
     ``edgar_fund``/``edgar_notes``/``edgar_read``/``edgar_document`` async
     handlers through fakes, replaying the documented first call and then
     cursor-only calls (constraints rule 7b) to the end of the page set.

Fast, no network: importing the tool modules only registers ``@tool``
decorators; the continuation-flow tests monkeypatch filing selection and
extraction so no request is made.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import pytest

from edgar.ai.mcp.server import SERVER_INSTRUCTIONS, _import_tools
from edgar.ai.mcp.tools.base import TOOLS
from edgar.ai.mcp.tools.prompts import PROMPTS, get_prompt

_import_tools()

ROOT = Path(__file__).parents[1]

# docs/ai/ was deleted (constraints rule 7c) -- MCP_QUICKSTART.md is the one
# surviving standalone doc source.
DOC_PATHS = (
    ROOT / "edgar/ai/mcp/docs/MCP_QUICKSTART.md",
)

MARKED_JSON = re.compile(
    r"<!--\s*MCP_TOOL_CALL_EXAMPLE\s*-->\s*```json\s*(.*?)\s*```",
    flags=re.DOTALL,
)

# One "Examples:" prose block, up to the next blank line or marked block.
PROSE_BLOCK = re.compile(r"Examples?:\s*\n(.*?)(?:\n\s*\n|<!--|\Z)", flags=re.DOTALL)

# key="value", key=["a","b"], key=true/false, or a bare key=123/12.5 number,
# anywhere in a prose bullet line.
KEY_VALUE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\[[^\]]*\]|true|false|-?\d+(?:\.\d+)?)')

# Tools excluded from prose-example checking, with the reason recorded here.
# A tool is never silently dropped -- if one belongs here, say why.
_PROSE_EXCLUDED_TOOLS: dict[str, str] = {}


def _tools_with_prose_examples_block() -> set[str]:
    """Every registered tool whose description has an "Examples:" block,
    minus ``_PROSE_EXCLUDED_TOOLS``.

    Derived from the live registry rather than a hardcoded allowlist, so a
    new tool that documents this style of example is covered automatically
    (a prior hardcoded list of 6 tools silently missed 8 others --
    edgar_company, edgar_compare, edgar_monitor, edgar_proxy,
    edgar_ownership, edgar_screen, edgar_trends, edgar_search -- that use
    the identical "- label: key=\"value\"" syntax).
    """
    return {
        name for name, info in TOOLS.items()
        if name not in _PROSE_EXCLUDED_TOOLS and PROSE_BLOCK.search(info["description"])
    }


# Tools whose description= uses the "Examples:\n- label: key=\"value\", ..."
# prose style, in addition to any marked JSON block. edgar_document has no
# such prose block (only marked JSON), so it is checked there only.
DESCRIPTION_EXAMPLE_TOOLS = _tools_with_prose_examples_block()


# =============================================================================
# Extraction: marked JSON
# =============================================================================


def _extract_examples(source: str, content: str) -> list[dict[str, Any]]:
    """Decode every marked JSON tool call in one documentation source."""
    examples = []
    for match in MARKED_JSON.finditer(content):
        try:
            example = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{source}: invalid marked tool-call JSON: {exc}") from exc
        assert isinstance(example, dict), f"{source}: marked tool call must be an object"
        assert set(example) == {"tool", "arguments"}, (
            f"{source}: marked tool call must contain only 'tool' and 'arguments'"
        )
        assert isinstance(example["tool"], str), f"{source}: tool name must be a string"
        assert isinstance(example["arguments"], dict), f"{source}: arguments must be an object"
        example["_source"] = source
        examples.append(example)
    return examples


# =============================================================================
# Extraction: prose "Examples:" bullets (P2-L9: not only marked JSON)
# =============================================================================


def _parse_prose_value(raw: str):
    """A quoted string, a ``["a","b"]`` list literal, ``true``/``false``, or
    a bare number (``2834``, ``5.0``), to a Python value."""
    if raw.startswith("["):
        inner = raw[1:-1]
        return [item.strip().strip('"') for item in inner.split(",") if item.strip()]
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith('"'):
        return raw[1:-1]
    return float(raw) if "." in raw else int(raw)


def _extract_prose_examples(tool_name: str, description: str) -> list[dict[str, Any]]:
    """Every "Examples:" bullet in a tool's description that carries at
    least one ``key="value"``/``key=[...]`` pair. Bullets that are plain
    narration (no such pair) are not examples and are skipped."""
    block_match = PROSE_BLOCK.search(description)
    if not block_match:
        return []

    examples = []
    for line in block_match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        pairs = KEY_VALUE.findall(line)
        if not pairs:
            continue
        arguments = {key: _parse_prose_value(value) for key, value in pairs}
        examples.append({
            "tool": tool_name,
            "arguments": arguments,
            "_source": f"{tool_name} description (prose): {line}",
        })
    return examples


# =============================================================================
# Schema + selector checks
# =============================================================================


def _check_value(path: str, value: Any, schema: dict[str, Any]) -> None:
    """Check enums and nested object fields from a JSON Schema property."""
    enum = schema.get("enum")
    if enum is not None:
        assert value in enum, f"{path}: {value!r} is not one of {enum}"

    if schema.get("type") == "array":
        assert isinstance(value, list), f"{path}: expected an array"
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _check_value(f"{path}[{index}]", item, item_schema)
    elif schema.get("type") == "object":
        assert isinstance(value, dict), f"{path}: expected an object"
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - set(value)
        assert not missing, f"{path}: missing required fields {sorted(missing)}"
        unknown = set(value) - set(properties)
        assert not unknown, f"{path}: unknown fields {sorted(unknown)}"
        for key, item in value.items():
            _check_value(f"{path}.{key}", item, properties[key])


def _has_filing_selector(tool_name: str, arguments: dict[str, Any]) -> bool:
    """A cursor alone always selects its own filing (constraints rule 7b)."""
    if arguments.get("cursor"):
        return True
    if tool_name == "edgar_filing":
        return bool(arguments.get("input")) or bool(
            arguments.get("identifier") and arguments.get("form")
        )
    if tool_name in {"edgar_read", "edgar_notes"}:
        return bool(arguments.get("accession_number") or arguments.get("identifier"))
    if tool_name == "edgar_document":
        return bool(arguments.get("accession_number") or arguments.get("url"))
    if tool_name == "edgar_fund" and arguments.get("action") in {
        "bdc_portfolio",
        "bdc_nonaccrual",
    }:
        return bool(arguments.get("accession_number") or arguments.get("identifier"))
    return True


def _check_conditional_selectors(tool_name: str, arguments: dict[str, Any], source: str) -> None:
    """Validate selector requirements that JSON Schema cannot express.

    A cursor is always enough on its own (constraints rule 7b): every other
    selector or filter it carries -- filing identity, topic/detail,
    section, document, query -- defaults to the cursor's own value when the
    call omits it, so a cursor example is never required to repeat them.
    """
    prefix = f"{source}: {tool_name}"
    action = arguments.get("action")

    if not _has_filing_selector(tool_name, arguments):
        raise AssertionError(f"{prefix} example must include a filing selector or a cursor")

    if tool_name == "edgar_filing":
        assert arguments.get("input") or (arguments.get("identifier") and arguments.get("form")), (
            f"{prefix} needs input or both identifier and form"
        )
    elif tool_name == "edgar_read":
        if not arguments.get("cursor"):
            if arguments.get("identifier"):
                assert arguments.get("form"), f"{prefix} needs form with identifier"
            if arguments.get("period"):
                assert arguments.get("identifier") and arguments.get("form"), (
                    f"{prefix} needs identifier and form with period"
                )
        sections = arguments.get("sections")
        if arguments.get("cursor") and sections is not None:
            assert isinstance(sections, list) and len(sections) == 1, (
                f"{prefix} cursor's sections, if given, must be exactly one section name"
            )
    elif tool_name == "edgar_notes":
        if not arguments.get("cursor") and arguments.get("period"):
            assert arguments.get("identifier") and arguments.get("form"), (
                f"{prefix} needs identifier and form with period"
            )
    elif tool_name == "edgar_fund":
        if action == "bdc_search":
            assert arguments.get("query"), f"{prefix} bdc_search needs query"
        if arguments.get("cursor"):
            assert action in {"bdc_portfolio", "bdc_nonaccrual"}, (
                f"{prefix} cursor requires a pageable BDC action"
            )
    elif tool_name == "edgar_document":
        if action == "search":
            assert arguments.get("query"), f"{prefix} search needs query"
        if action == "read" and not arguments.get("cursor"):
            assert arguments.get("document") or arguments.get("url"), (
                f"{prefix} read needs an exact document or SEC document URL"
            )


def _check_source_examples(source: str, examples: list[dict[str, Any]]) -> None:
    """Validate every example's parameter names, enum values and selector."""
    for example in examples:
        tool_name = example["tool"]
        arguments = example["arguments"]
        assert tool_name in TOOLS, f"{source}: '{tool_name}' is not a registered MCP tool"
        schema = TOOLS[tool_name]["schema"]
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - set(arguments)
        assert not missing, f"{source}: {tool_name} missing required fields {sorted(missing)}"
        unknown = set(arguments) - set(properties)
        assert not unknown, f"{source}: {tool_name} uses unknown fields {sorted(unknown)}"
        for key, value in arguments.items():
            _check_value(f"{source}: {tool_name}.{key}", value, properties[key])
        _check_conditional_selectors(tool_name, arguments, source)


# =============================================================================
# Gather every documented example
# =============================================================================


def _all_examples() -> dict[str, list[dict[str, Any]]]:
    """Every marked-JSON and prose example, keyed by source label.

    Only sources that document at least one example carry a key. A tool
    without ANY parsed example (no marked JSON and no "Examples:" block at
    all, e.g. edgar_monitor's "- All latest: (no parameters)" bullet, which
    documents an empty call rather than a key="value" one) is intentionally
    out of scope -- but ``DESCRIPTION_EXAMPLE_TOOLS`` is derived from the
    registry (``_tools_with_prose_examples_block``), not hardcoded, and
    ``test_every_tool_with_an_examples_block_yields_parsed_examples`` fails
    if a tool that DOES have an "Examples:" block yields zero.
    """
    examples: dict[str, list[dict[str, Any]]] = {}

    for path in DOC_PATHS:
        label = str(path.relative_to(ROOT))
        examples[label] = _extract_examples(label, path.read_text())

    for name, info in TOOLS.items():
        marked = _extract_examples(f"{name} description", info["description"])
        prose = (
            _extract_prose_examples(name, info["description"])
            if name in DESCRIPTION_EXAMPLE_TOOLS
            else []
        )
        found = marked + prose
        if found:
            examples[f"{name} description"] = found

    return examples


@pytest.mark.fast
def test_documented_examples_match_registered_schemas():
    """Every marked JSON and prose example, across every surviving doc
    source, uses parameter names/enum values the registered tool accepts,
    and a valid filing selector or cursor."""
    all_examples = _all_examples()

    for source, examples in all_examples.items():
        assert examples, f"{source}: no documented tool-call examples found"
        _check_source_examples(source, examples)

    # Every tool that carries a documented example is a real registered
    # tool with at least one example checked -- guards against silently
    # narrowing coverage back down to a subset of tools.
    for name in DESCRIPTION_EXAMPLE_TOOLS:
        assert all_examples[f"{name} description"], f"{name}: expected documented examples"


@pytest.mark.fast
def test_every_tool_with_an_examples_block_yields_parsed_examples():
    """A registered tool whose description has an "Examples:" block must
    yield at least one parsed ``key="value"`` bullet, not zero.

    ``DESCRIPTION_EXAMPLE_TOOLS`` (``_tools_with_prose_examples_block``) is
    the set this checks -- it is derived from the live registry, so this
    guard fires the moment a NEW tool's "Examples:" block goes unparsed
    (a bug in the bullet syntax, or a regression in ``KEY_VALUE``/
    ``PROSE_BLOCK``), not just when today's known set does. This is the
    exact class of bug the fix-round finding named: a hardcoded 6-tool
    allowlist silently missed edgar_company/edgar_compare/edgar_monitor/
    edgar_proxy/edgar_ownership/edgar_screen/edgar_trends/edgar_search,
    which all have a checkable "Examples:" block.
    """
    tools_with_examples_block = _tools_with_prose_examples_block()
    assert tools_with_examples_block, "expected at least one tool with an Examples: block"

    for name in sorted(tools_with_examples_block):
        examples = _extract_prose_examples(name, TOOLS[name]["description"])
        assert examples, (
            f"{name}: has an 'Examples:' block but yielded zero parsed examples "
            "-- check the bullet syntax against KEY_VALUE/PROSE_BLOCK"
        )


@pytest.mark.fast
def test_report_example_counts():
    """Not an assertion -- prints how many examples were checked per
    source, for the task report (brief item 1: report before/after)."""
    all_examples = _all_examples()
    total = sum(len(examples) for examples in all_examples.values())
    print(f"\nDocumented examples checked: {total} across {len(all_examples)} sources")
    for source, examples in sorted(all_examples.items()):
        print(f"  {source}: {len(examples)}")


# =============================================================================
# Self-checks (P2-L9: restore the checker's own tests, against
# _check_source_examples specifically)
# =============================================================================


@pytest.mark.fast
def test_checker_rejects_unknown_parameter():
    source = "self-check"
    examples = [{
        "tool": "edgar_read",
        "arguments": {"identifier": "ARCC", "form": "10-Q", "sectionsx": ["mda"]},
        "_source": source,
    }]
    with pytest.raises(AssertionError, match="unknown fields"):
        _check_source_examples(source, examples)


@pytest.mark.fast
def test_checker_rejects_invalid_enum_value():
    source = "self-check"
    examples = [{
        "tool": "edgar_notes",
        "arguments": {"identifier": "ARCC", "form": "10-Q", "detail": "extreme"},
        "_source": source,
    }]
    with pytest.raises(AssertionError, match="is not one of"):
        _check_source_examples(source, examples)


@pytest.mark.fast
def test_checker_rejects_example_missing_filing_selector():
    """A call with no cursor and no filing identity is not a valid
    example -- unlike a cursor-only call, which now IS valid (rule 7b)."""
    source = "self-check"
    examples = [{
        "tool": "edgar_read",
        "arguments": {"sections": ["mda"]},
        "_source": source,
    }]
    with pytest.raises(AssertionError, match="filing selector"):
        _check_source_examples(source, examples)


@pytest.mark.fast
def test_checker_accepts_cursor_alone_as_a_filing_selector():
    """The mirror image of the previous test: a cursor alone is a valid
    filing selector under constraints rule 7b."""
    source = "self-check"
    examples = [{
        "tool": "edgar_read",
        "arguments": {"cursor": "<section_pages.mda.next_cursor>"},
        "_source": source,
    }]
    _check_source_examples(source, examples)  # must not raise


# =============================================================================
# P2-L8: tool count in the docs matches the registered count
# =============================================================================


@pytest.mark.fast
def test_server_instructions_tool_count_matches_registered_tools():
    match = re.search(r"(\d+) tools", SERVER_INSTRUCTIONS)
    assert match, "SERVER_INSTRUCTIONS should state the tool count"
    assert int(match.group(1)) == len(TOOLS)


@pytest.mark.fast
def test_quickstart_tool_count_matches_registered_tools():
    text = (ROOT / "edgar/ai/mcp/docs/MCP_QUICKSTART.md").read_text()
    match = re.search(r"registers (\d+) tools", text)
    assert match, "MCP_QUICKSTART.md should state the registered tool count"
    assert int(match.group(1)) == len(TOOLS)


@pytest.mark.fast
def test_no_stale_tool_count_elsewhere_in_the_repo():
    """P2-L8's exact regression: a stale '13 tools' (or any count other
    than the registered one) left over anywhere still in the repo."""
    stale = []
    for base in (ROOT / "edgar/ai/mcp", ROOT / "edgar/ai/skills"):
        for path in base.rglob("*"):
            if path.suffix not in {".md", ".py", ".yaml", ".yml"} or not path.is_file():
                continue
            text = path.read_text(errors="ignore")
            for match in re.finditer(r"\b(\d+)\s+tools\b", text):
                if int(match.group(1)) != len(TOOLS):
                    stale.append(f"{path.relative_to(ROOT)}: '{match.group(0)}'")
    assert not stale, f"stale tool-count text found: {stale}"


# =============================================================================
# prompts.py: tool names mentioned in rendered prompt text are real tools
# =============================================================================


def _minimal_prompt_arguments(name: str) -> dict[str, str]:
    return {arg.name: f"TEST-{arg.name}" for arg in PROMPTS[name].arguments if arg.required}


@pytest.mark.fast
@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_prompt_text_only_mentions_real_tools(name):
    result = get_prompt(name, _minimal_prompt_arguments(name))
    text = result.messages[0].content.text
    mentioned = set(re.findall(r"\bedgar_[a-z_]+\b", text))
    unknown = mentioned - set(TOOLS)
    assert not unknown, f"prompt '{name}' mentions unregistered tool(s): {sorted(unknown)}"


# =============================================================================
# Skill YAMLs that contain MCP tool calls (edgar_xxx(key="value", ...))
# =============================================================================

SKILL_CALL = re.compile(r"(edgar_\w+)\(([^()]*)\)", flags=re.DOTALL)


@pytest.mark.fast
def test_skill_yaml_mcp_calls_use_valid_tools_and_parameters():
    skills_dir = ROOT / "edgar/ai/skills"
    checked = 0
    for path in sorted(skills_dir.rglob("*.yaml")):
        text = path.read_text()
        for tool_match in SKILL_CALL.finditer(text):
            tool_name, args_text = tool_match.group(1), tool_match.group(2)
            source = f"{path.relative_to(ROOT)}: {tool_match.group(0)[:60]}"
            assert tool_name in TOOLS, f"{source}: '{tool_name}' is not a registered MCP tool"
            properties = TOOLS[tool_name]["schema"]["properties"]
            for key, raw_value in KEY_VALUE.findall(args_text):
                assert key in properties, (
                    f"{source}: unknown parameter '{key}' (known: {sorted(properties)})"
                )
                enum = properties[key].get("enum")
                if enum is not None:
                    value = _parse_prose_value(raw_value)
                    assert value in enum, f"{source}: {key}={value!r} is not one of {enum}"
            checked += 1
    assert checked > 0, "expected at least one MCP call in the skill YAMLs"


# =============================================================================
# Item 5: the documented multi-call continuation flows actually run.
#
# Each test replays the tool's OWN first documented marked-JSON example
# (asserted below to still be a subset of what the test calls with) against
# fakes, then continues with cursor-only calls (matching the tool's
# documented cursor-only continuation example) until next_cursor is null.
# =============================================================================


def _first_documented_example(tool_name: str, action: Optional[str] = None) -> dict[str, Any]:
    examples = _extract_examples(f"{tool_name} description", TOOLS[tool_name]["description"])
    non_cursor = [ex for ex in examples if "cursor" not in ex["arguments"]]
    if action is not None:
        non_cursor = [ex for ex in non_cursor if ex["arguments"].get("action") == action]
    assert non_cursor, f"{tool_name}: expected a non-cursor marked example (action={action!r})"
    return non_cursor[0]["arguments"]


def _documented_cursor_example_keys(tool_name: str) -> set[str]:
    examples = _extract_examples(f"{tool_name} description", TOOLS[tool_name]["description"])
    cursor_examples = [ex for ex in examples if "cursor" in ex["arguments"]]
    assert cursor_examples, f"{tool_name}: expected a cursor marked example"
    return set(cursor_examples[0]["arguments"])


@pytest.mark.fast
@pytest.mark.asyncio
class TestDocumentedContinuationFlows:
    async def test_edgar_fund_bdc_portfolio_cursor_only_flow(self, monkeypatch):
        from dataclasses import dataclass

        from edgar.ai.mcp.tools import continuation
        from edgar.ai.mcp.tools.bdc.identity import BdcLookup
        from edgar.ai.mcp.tools.continuation import ResultCache
        from edgar.ai.mcp.tools.fund import edgar_fund
        from edgar.ai.mcp.tools.selection import FilingSelection
        from edgar.bdc.investments import PortfolioInvestment, PortfolioInvestments

        monkeypatch.setattr(continuation, "results_cache", ResultCache(max_entries=8))
        monkeypatch.setattr(continuation, "text_cache", ResultCache(max_entries=8, max_bytes=1024 * 1024))

        # The doc's first bdc_portfolio example must only use keys this
        # flow actually drives (guards against the doc and this test
        # drifting apart).
        doc_first = _first_documented_example("edgar_fund")
        assert doc_first["action"] == "bdc_portfolio"
        assert set(doc_first) <= {"action", "identifier", "form", "period", "borrower", "limit"}
        assert _documented_cursor_example_keys("edgar_fund") == {"action", "cursor"}

        @dataclass
        class _FakeBDC:
            cik: int = 1287750
            name: str = "ARES CAPITAL CORP"
            state: str = "MD"
            is_active: bool = True
            report_year: Optional[int] = 2025

            def get_company(self):
                return type("C", (), {"cik": self.cik})()

        class _FakeFiling:
            accession_number = "0001628280-26-050307"
            cik = 1287750
            form = "10-Q"
            report_date = "2026-06-30"
            filing_date = "2026-08-01"
            company = "ARES CAPITAL CORP"
            homepage_url = "https://www.sec.gov/cgi-bin/browse-edgar"

            def xbrl(self):
                return None

        monkeypatch.setattr(
            "edgar.ai.mcp.tools.selection.resolve_report_filing",
            lambda **kwargs: FilingSelection(filing=_FakeFiling(), selected_by="latest"),
        )
        monkeypatch.setattr(
            "edgar.ai.mcp.tools.bdc.identity.resolve_bdc",
            lambda identifier: BdcLookup(bdc=_FakeBDC()),
        )

        investments = PortfolioInvestments([
            PortfolioInvestment(
                identifier=f"Ivy Hill Fund {i}", company_name=f"Ivy Hill Fund {i}",
                investment_type="First lien senior secured loan",
            )
            for i in range(25)
        ], extraction_method="xbrl_facts")
        monkeypatch.setattr(
            "edgar.bdc.investments.portfolio_investments_from_filing",
            lambda filing, include_untyped=False, xbrl=None: investments,
        )

        first = await edgar_fund(**doc_first)
        assert first.success is True, first.error
        collected = [inv["identifier"] for inv in first.data["investments"]]
        cursor = first.data["page"]["next_cursor"]
        assert cursor is not None

        hops = 0
        while cursor:
            hops += 1
            page = await edgar_fund(action="bdc_portfolio", cursor=cursor)
            assert page.success is True, page.error
            collected.extend(inv["identifier"] for inv in page.data["investments"])
            cursor = page.data["page"]["next_cursor"]

        assert hops >= 1, "expected at least one cursor-only continuation hop"
        assert collected == [f"Ivy Hill Fund {i}" for i in range(25)]

    async def test_edgar_notes_cursor_only_flow(self, monkeypatch):
        from edgar.ai.mcp.tools import continuation
        from edgar.ai.mcp.tools.continuation import ResultCache
        from edgar.ai.mcp.tools.notes import edgar_notes
        from edgar.ai.mcp.tools.selection import FilingSelection

        monkeypatch.setattr(continuation, "results_cache", ResultCache(max_entries=8))
        monkeypatch.setattr(continuation, "text_cache", ResultCache(max_entries=8, max_bytes=1024 * 1024))

        doc_first = _first_documented_example("edgar_notes")
        assert set(doc_first) <= {"topic", "identifier", "form", "period", "detail", "limit"}
        assert _documented_cursor_example_keys("edgar_notes") == {"cursor"}

        from edgar.ai.mcp.tools.continuation import TEXT_PAGE_CHARS

        context_text = "\n".join(f"Debt disclosure line {i}." for i in range(600))
        assert len(context_text) > TEXT_PAGE_CHARS * 2

        class _FakeNote:
            number = 1
            title = "Debt"
            expands: list = []
            expands_statements: list = []
            tables: list = []

            def to_context(self, detail="standard"):
                return context_text

        class _FakeNotes(list):
            def search(self, keyword):
                return [n for n in self if keyword.lower() in n.title.lower()]

            def __getitem__(self, key):
                if isinstance(key, int):
                    for n in self:
                        if n.number == key:
                            return n
                    return None
                return super().__getitem__(key)

        note = _FakeNote()
        fake_notes = _FakeNotes([note])

        class _FakeReportObj:
            period_of_report = "2026-06-30"
            notes_obj = fake_notes

            @property
            def notes(self):
                return self.notes_obj

        class _FakeFiling:
            accession_number = "0001213900-26-090000"
            form = "10-Q"
            cik = 845385
            company = "Princeton Capital Corp"
            filing_date = "2026-08-01"
            homepage_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            report_date = "2026-06-30"

            def obj(self):
                return _FakeReportObj()

        monkeypatch.setattr(
            "edgar.ai.mcp.tools.selection.resolve_report_filing",
            lambda **kwargs: FilingSelection(filing=_FakeFiling(), selected_by="latest"),
        )

        first = await edgar_notes(**doc_first)
        assert first.success is True, first.error
        note_data = first.data["notes"][0]
        collected = note_data["context"]
        cursor = note_data["context_page"]["next_cursor"]
        assert cursor is not None

        hops = 0
        while cursor:
            hops += 1
            page = await edgar_notes(cursor=cursor)
            assert page.success is True, page.error
            collected += page.data["context"]
            cursor = page.data["page"]["next_cursor"]

        assert hops >= 1
        assert collected == context_text

    async def test_edgar_read_cursor_only_flow(self, monkeypatch):
        from edgar.ai.mcp.tools import reader as reader_module
        from edgar.ai.mcp.tools.reader import edgar_read
        from edgar.ai.mcp.tools.selection import FilingSelection

        doc_first = _first_documented_example("edgar_read")
        assert set(doc_first) <= {"identifier", "form", "period", "sections"}
        assert _documented_cursor_example_keys("edgar_read") == {"cursor"}

        from edgar.ai.mcp.tools.continuation import TEXT_PAGE_CHARS

        section_text = "\n".join(f"MD&A paragraph {i}." for i in range(700))
        assert len(section_text) > TEXT_PAGE_CHARS * 2

        class _FakeFiling:
            accession_number = "0001628280-26-050307"
            form = "10-Q"
            cik = 1287750
            company = "ARES CAPITAL CORP"
            filing_date = "2026-08-01"
            homepage_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            report_date = "2026-06-30"

            def obj(self):
                return object()

        monkeypatch.setattr(
            "edgar.ai.mcp.tools.selection.resolve_report_filing",
            lambda **kwargs: FilingSelection(filing=_FakeFiling(), selected_by="latest"),
        )
        monkeypatch.setattr(reader_module, "_extract_section", lambda obj, form, section: section_text)

        first = await edgar_read(**doc_first)
        assert first.success is True, first.error
        collected = first.data["sections"]["mda"]
        cursor = first.data["section_pages"]["mda"]["next_cursor"]
        assert cursor is not None

        hops = 0
        while cursor:
            hops += 1
            page = await edgar_read(cursor=cursor)
            assert page.success is True, page.error
            collected += page.data["sections"]["mda"]
            cursor = page.data["section_pages"]["mda"]["next_cursor"]

        assert hops >= 1
        assert collected == section_text

    async def test_edgar_document_read_cursor_only_flow(self, monkeypatch):
        from edgar._filings import Filing
        from edgar.ai.mcp.tools import continuation
        from edgar.ai.mcp.tools.continuation import ResultCache
        from edgar.ai.mcp.tools.document import edgar_document

        monkeypatch.setattr(continuation, "text_cache", ResultCache(max_entries=16, max_bytes=2_000_000))
        monkeypatch.setattr(continuation, "results_cache", ResultCache(max_entries=16))

        doc_first = _first_documented_example("edgar_document", action="read")
        # This tool's doc uses a placeholder filename ("[exact-filename-from-list]"),
        # which is not a real attachment identity -- substitute a concrete
        # one for execution, keeping the same key shape the doc documents.
        assert doc_first["action"] == "read"
        assert set(doc_first) <= {"action", "accession_number", "document"}
        assert _documented_cursor_example_keys("edgar_document") == {"action", "cursor"}

        accession = "0001628280-26-050307"
        attachment_text = ("0123456789" * 1_300) + "tail"

        class _FakeAttachment:
            sequence_number = "5"
            document = "agreement-a.htm"
            document_type = "EX-10.1"
            description = "Loan agreement"
            display_description = "Loan agreement"
            size = len(attachment_text)
            ixbrl = False
            path = f"/Archives/edgar/data/1287750/{accession}/agreement-a.htm"

            @property
            def url(self):
                return f"https://www.sec.gov{self.path}"

            @property
            def empty(self):
                return False

            @property
            def extension(self):
                return ".htm"

            def is_text(self):
                return True

            def is_html(self):
                return True

            def is_binary(self):
                return False

            def markdown(self):
                return attachment_text

            def text(self):
                return attachment_text

        class LocalFiling(Filing):
            def __repr__(self):
                return f"LocalFiling({self.accession_number})"

        filing = LocalFiling(cik=1287750, company="ARES CAPITAL CORP", form="10-Q",
                              filing_date="2026-08-01", accession_no=accession)
        attachments = [_FakeAttachment()]
        monkeypatch.setattr(Filing, "attachments", property(lambda self: attachments))
        monkeypatch.setattr("edgar.find", lambda **kwargs: filing)

        call = {**doc_first, "document": "agreement-a.htm"}
        first = await edgar_document(**call)
        assert first.success is True, first.error
        collected = first.data["text"]
        cursor = first.data["page"]["next_cursor"]
        assert cursor is not None

        hops = 0
        while cursor:
            hops += 1
            page = await edgar_document(action="read", cursor=cursor)
            assert page.success is True, page.error
            collected += page.data["text"]
            cursor = page.data["page"]["next_cursor"]

        assert hops >= 1
        assert collected == attachment_text
