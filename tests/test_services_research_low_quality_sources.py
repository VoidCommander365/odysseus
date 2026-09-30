"""ResearchHandler._extract_sources must gate low-quality findings.

The authoritative src/research_handler.py filters findings whose summary is
junk boilerplate (via research_utils.is_low_quality) before listing them as
cited sources, so "the page does not contain relevant information" URLs do
not show up as sources, and a junk finding seen first does not suppress the
good title for the same URL.
"""

from src.research_handler import ResearchHandler

JUNK = "The page does not contain relevant information"


def test_low_quality_summary_is_not_a_source():
    out = ResearchHandler._extract_sources([{"url": "http://a", "title": "T", "summary": JUNK}])
    assert out == []


def test_good_summary_is_kept():
    out = ResearchHandler._extract_sources(
        [{"url": "http://a", "title": "T", "summary": "Detailed statistics about the topic"}]
    )
    assert out == [{"url": "http://a", "title": "T"}]


def test_junk_first_no_longer_suppresses_the_good_finding():
    out = ResearchHandler._extract_sources(
        [
            {"url": "http://a", "title": "Bad", "summary": JUNK},
            {"url": "http://a", "title": "Good", "summary": "Real data about the topic"},
        ]
    )
    assert out == [{"url": "http://a", "title": "Good"}]


def test_evidence_is_checked_when_summary_missing():
    out = ResearchHandler._extract_sources(
        [{"url": "http://a", "title": "T", "evidence": "Concrete evidence text"}]
    )
    assert out == [{"url": "http://a", "title": "T"}]
