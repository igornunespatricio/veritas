"""Domain events for agent communication."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class DomainEvent:
    """Base domain event for agent communication."""

    event_id: str
    timestamp: datetime
    correlation_id: str
    event_type: str
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> "DomainEvent":
        """Factory method to create a domain event."""
        return cls(
            event_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id or str(uuid4()),
            event_type=event_type,
            payload=payload,
        )


@dataclass
class ResearchCompleted(DomainEvent):
    """Event when researcher completes research."""

    topic: str
    sources: list[dict[str, str]]
    findings: list[str]

    @classmethod
    def create(
        cls,
        topic: str,
        sources: list[dict[str, str]],
        findings: list[str],
        correlation_id: str | None = None,
    ) -> "ResearchCompleted":
        """Factory method to create a research completed event."""
        return cls(
            event_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id or str(uuid4()),
            event_type="research.completed",
            payload={
                "topic": topic,
                "sources": sources,
                "findings": findings,
            },
            topic=topic,
            sources=sources,
            findings=findings,
        )


@dataclass
class FactCheckCompleted(DomainEvent):
    """Event when fact-checker completes verification."""

    claims: list[dict[str, Any]]
    verified_claims: list[dict[str, Any]]
    confidence_scores: dict[str, float]

    @classmethod
    def create(
        cls,
        claims: list[dict[str, Any]],
        verified_claims: list[dict[str, Any]],
        confidence_scores: dict[str, float],
        correlation_id: str | None = None,
    ) -> "FactCheckCompleted":
        """Factory method to create a fact-check completed event."""
        return cls(
            event_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id or str(uuid4()),
            event_type="fact_check.completed",
            payload={
                "claims": claims,
                "verified_claims": verified_claims,
                "confidence_scores": confidence_scores,
            },
            claims=claims,
            verified_claims=verified_claims,
            confidence_scores=confidence_scores,
        )


@dataclass
class SynthesisCompleted(DomainEvent):
    """Event when synthesizer completes merging insights."""

    insights: list[str]
    resolved_contradictions: list[dict[str, Any]]

    @classmethod
    def create(
        cls,
        insights: list[str],
        resolved_contradictions: list[dict[str, Any]],
        correlation_id: str | None = None,
    ) -> "SynthesisCompleted":
        """Factory method to create a synthesis completed event."""
        return cls(
            event_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id or str(uuid4()),
            event_type="synthesis.completed",
            payload={
                "insights": insights,
                "resolved_contradictions": resolved_contradictions,
            },
            insights=insights,
            resolved_contradictions=resolved_contradictions,
        )


@dataclass
class ReportWritten(DomainEvent):
    """Event when writer completes report."""

    title: str
    content: str
    format: str

    @classmethod
    def create(
        cls,
        title: str,
        content: str,
        format: str = "markdown",
        correlation_id: str | None = None,
    ) -> "ReportWritten":
        """Factory method to create a report written event."""
        return cls(
            event_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id or str(uuid4()),
            event_type="report.written",
            payload={
                "title": title,
                "content": content,
                "format": format,
            },
            title=title,
            content=content,
            format=format,
        )


@dataclass
class ReportReviewed(DomainEvent):
    """Event when critic completes review."""

    suggestions: list[str]
    score: float
    approved: bool

    @classmethod
    def create(
        cls,
        suggestions: list[str],
        score: float,
        approved: bool,
        correlation_id: str | None = None,
    ) -> "ReportReviewed":
        """Factory method to create a report reviewed event."""
        return cls(
            event_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id or str(uuid4()),
            event_type="report.reviewed",
            payload={
                "suggestions": suggestions,
                "score": score,
                "approved": approved,
            },
            suggestions=suggestions,
            score=score,
            approved=approved,
        )
