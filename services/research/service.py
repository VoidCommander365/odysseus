# services/research/service.py
"""Research service — deep research with LLM-in-the-loop."""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

from src.research_handler import ResearchHandler


@dataclass
class ResearchSource:
    """A source found during research."""
    url: str
    title: str
    snippet: str
    relevance: float = 0.0


@dataclass
class ResearchResult:
    """Result of a deep research query."""
    query: str
    summary: str
    sources: List[ResearchSource] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    tokens_used: int = 0
    duration_seconds: float = 0.0


class ResearchService:
    """
    Deep research service.

    Usage:
        service = ResearchService()
        result = await service.research("quantum computing advances 2024")
        print(result.summary)
    """

    def __init__(self):
        self.handler = ResearchHandler()
        self._active: dict = {}

    async def research(
        self,
        topic: str,
        llm_endpoint: str,
        llm_model: str,
        max_time: int = 300,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> ResearchResult:
        """
        Perform deep research on a topic.

        Args:
            topic: Research topic/question
            llm_endpoint: LLM API endpoint
            llm_model: Model to use
            max_time: Maximum time in seconds
            on_progress: Optional progress callback

        Returns:
            ResearchResult with findings
        """
        import time
        start = time.time()

        # Local task entry so call_research_service can store the
        # DeepResearcher instance (and raw report/stats) on it.
        task_entry: dict = {}

        result = await self.handler.call_research_service(
            topic,
            llm_endpoint,
            llm_model,
            max_time=max_time,
            progress_callback=on_progress,
            _task_entry=task_entry,
        )

        duration = time.time() - start

        # call_research_service returns a formatted markdown report string
        # (see ResearchHandler.call_research_service -> _format_research_report),
        # not a dict. Treat it as such; tolerate an unexpected dict/None defensively.
        if isinstance(result, dict):
            sources = [
                ResearchSource(
                    url=s.get("url", ""),
                    title=s.get("title", ""),
                    snippet=s.get("snippet", ""),
                    relevance=s.get("relevance", 0.0),
                )
                for s in result.get("sources", [])
                if isinstance(s, dict)
            ]
            return ResearchResult(
                query=topic,
                summary=result.get("summary", result.get("answer", "")),
                sources=sources,
                sections=result.get("sections", []),
                tokens_used=result.get("tokens_used", 0),
                duration_seconds=duration,
            )

        report = result if isinstance(result, str) else ""

        # Sources come from the DeepResearcher's findings via the canonical
        # extractor (same path as ResearchHandler.get_sources), not from
        # markdown parsing — the formatted report no longer embeds a
        # "### Sources" section.
        sources: List[ResearchSource] = []
        researcher = task_entry.get("researcher")
        if researcher is not None and getattr(researcher, "findings", None):
            for s in ResearchHandler._extract_sources(researcher.findings):
                # _extract_sources yields {url, title, image?}; the canonical
                # extractor does not expose snippets, so default to empty.
                sources.append(
                    ResearchSource(url=s.get("url", ""), title=s.get("title", ""), snippet="")
                )

        return ResearchResult(
            query=topic,
            summary=report,
            sources=sources,
            duration_seconds=duration,
        )

    def start_background(
        self,
        session_id: str,
        topic: str,
        llm_endpoint: str,
        llm_model: str,
        max_time: int = 300,
    ) -> dict:
        """Start research in background. Returns task info."""
        return self.handler.start_research(
            session_id, topic, llm_endpoint, llm_model, max_time
        )

    def get_status(self, session_id: str) -> Optional[dict]:
        """Get status of background research."""
        return self.handler.get_status(session_id)

    def cancel(self, session_id: str) -> bool:
        """Cancel background research."""
        return self.handler.cancel_research(session_id)
