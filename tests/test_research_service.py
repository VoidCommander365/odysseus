"""Tests for ResearchService — correct handling of the handler's string report.

ResearchHandler.call_research_service returns a *formatted markdown string*,
not a dict. ResearchService.research() must consume that contract without
raising (the previous code called ``.get()`` on the string and blew up on
every successful research call).

Sources are not parsed from the markdown (the formatted report no longer
embeds a "### Sources" section). Instead, ResearchService passes a local
task entry to call_research_service, reads the DeepResearcher stored on it,
and runs the canonical ResearchHandler._extract_sources over its findings.
"""

import asyncio

import pytest

from services.research.service import (
    ResearchService,
    ResearchResult,
    ResearchSource,
)


# A faithful slice of what ResearchHandler._format_research_report emits
# (markdown only — no "### Sources" section).
SAMPLE_REPORT = """---

## Research Summary

**Duration:** 12.3s | **Rounds:** 3 | **Queries:** 5 | **URLs Analyzed:** 7

---

# Findings

Quantum error correction saw major advances in 2024. See [an inline note](https://inline.example/not-a-source) here.

---

**The AI has analyzed all research findings above.**
"""


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _StubResearcher:
    """Stands in for DeepResearcher; exposes a findings list."""

    def __init__(self, findings):
        self.findings = findings


class _StubHandler:
    """Stands in for ResearchHandler; returns a string like the real one and
    stores the researcher on the task entry, like the real one does."""

    def __init__(self, report, findings=None):
        self._report = report
        self._findings = findings
        self.called_with = None

    async def call_research_service(self, topic, llm_endpoint, llm_model,
                                    max_time=300, progress_callback=None,
                                    _task_entry=None):
        self.called_with = (topic, llm_endpoint, llm_model, max_time)
        if _task_entry is not None and self._findings is not None:
            _task_entry["researcher"] = _StubResearcher(self._findings)
        return self._report


class TestResearchOnStringReport:
    def _service(self, report, findings=None):
        svc = ResearchService()
        svc.handler = _StubHandler(report, findings)
        return svc

    def test_does_not_raise_on_string_report(self):
        svc = self._service(SAMPLE_REPORT)
        result = _run(svc.research("quantum", "http://llm", "model"))
        assert isinstance(result, ResearchResult)

    def test_summary_is_the_report(self):
        svc = self._service(SAMPLE_REPORT)
        result = _run(svc.research("quantum", "http://llm", "model"))
        assert "Quantum error correction" in result.summary
        assert result.query == "quantum"

    def test_sources_from_researcher_findings(self):
        findings = [
            {"url": "https://example.com/surface-codes", "title": "Surface Codes Paper",
             "summary": "Detailed statistics about the topic"},
            {"url": "https://example.com/lab", "title": "Lab Announcement",
             "summary": "Real data about the topic"},
            {"url": "https://example.com/surface-codes", "title": "Surface Codes Paper",
             "summary": "Detailed statistics about the topic"},
        ]
        svc = self._service(SAMPLE_REPORT, findings)
        result = _run(svc.research("quantum", "http://llm", "model"))
        urls = [s.url for s in result.sources]
        assert urls == [
            "https://example.com/surface-codes",
            "https://example.com/lab",
        ]
        assert all(isinstance(s, ResearchSource) for s in result.sources)

    def test_low_quality_findings_are_gated(self):
        findings = [
            {"url": "https://junk.example", "title": "Junk",
             "summary": "The page does not contain relevant information"},
            {"url": "https://good.example", "title": "Good",
             "summary": "Useful content here"},
        ]
        svc = self._service(SAMPLE_REPORT, findings)
        result = _run(svc.research("quantum", "http://llm", "model"))
        urls = [s.url for s in result.sources]
        assert urls == ["https://good.example"]

    def test_inline_links_in_report_are_not_sources(self):
        svc = self._service(SAMPLE_REPORT)
        result = _run(svc.research("quantum", "http://llm", "model"))
        urls = [s.url for s in result.sources]
        assert "https://inline.example/not-a-source" not in urls

    def test_duration_recorded(self):
        svc = self._service(SAMPLE_REPORT)
        result = _run(svc.research("quantum", "http://llm", "model"))
        assert result.duration_seconds >= 0.0

    def test_empty_report_yields_no_sources(self):
        svc = self._service("")
        result = _run(svc.research("quantum", "http://llm", "model"))
        assert result.sources == []
        assert result.summary == ""

    def test_source_contract_snippet_and_relevance(self):
        findings = [
            {"url": "https://example.com/x", "title": "X",
             "summary": "Detailed statistics about the topic"},
        ]
        svc = self._service(SAMPLE_REPORT, findings)
        result = _run(svc.research("quantum", "http://llm", "model"))
        assert len(result.sources) == 1
        source = result.sources[0]
        assert source.url == "https://example.com/x"
        assert source.title == "X"
        assert source.snippet == ""
        assert source.relevance == 0.0


class TestDictBackCompat:
    """A handler that returns a dict (legacy shape) must still work."""

    def test_dict_result_still_parsed(self):
        svc = ResearchService()

        class _DictHandler:
            async def call_research_service(self, *a, **k):
                return {
                    "summary": "done",
                    "sources": [
                        {"url": "https://x.example", "title": "X",
                         "snippet": "s", "relevance": 0.9},
                        "bad source row",
                    ],
                    "sections": ["intro"],
                    "tokens_used": 42,
                }

        svc.handler = _DictHandler()
        result = _run(svc.research("q", "http://llm", "model"))
        assert result.summary == "done"
        assert result.tokens_used == 42
        assert result.sections == ["intro"]
        assert result.sources[0].url == "https://x.example"
        assert result.sources[0].relevance == 0.9
