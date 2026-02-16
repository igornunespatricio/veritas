"""Pydantic response models for the API."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.orchestration.workflow import WorkflowStage


class JobStatus(str, Enum):
    """Status of a research job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchSource(BaseModel):
    """Source information from research."""

    title: str
    url: str


class FactCheckClaim(BaseModel):
    """A claim with verification status."""

    claim: str
    status: str
    confidence: float | None = None
    notes: str | None = None


class ResearchStatusResponse(BaseModel):
    """Response model for job status check."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    topic: str = Field(..., description="Research topic")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    # Progress info (when processing)
    current_stage: str | None = Field(
        default=None, description="Current workflow stage"
    )
    progress_percentage: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Progress percentage",
    )

    # Result info (when completed)
    error: str | None = Field(default=None, description="Error message if failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "123e4567-e89b-12d3-a456-426614174000",
                    "status": "processing",
                    "topic": "What are the environmental impacts of electric vehicles?",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:35:00Z",
                    "current_stage": "fact_check",
                    "progress_percentage": 40,
                }
            ]
        }
    }


class ResearchJobResponse(ResearchStatusResponse):
    """Full response model with research results."""

    # Research stage results
    sources: list[ResearchSource] | None = Field(
        default=None,
        description="List of sources found during research",
    )
    findings: list[str] | None = Field(
        default=None,
        description="Key findings from research",
    )

    # Fact-check results
    claims_verified: int | None = Field(
        default=None,
        description="Number of claims verified",
    )
    claims_partially_verified: int | None = Field(
        default=None,
        description="Number of claims partially verified",
    )
    claims_disputed: int | None = Field(
        default=None,
        description="Number of claims disputed",
    )
    claims_unverified: int | None = Field(
        default=None,
        description="Number of claims unverified",
    )

    # Synthesis results
    insights: list[Any] | None = Field(
        default=None,
        description="Synthesized insights (can be strings or dicts)",
    )

    # Report results
    report_title: str | None = Field(
        default=None,
        description="Generated report title",
    )
    report_content: str | None = Field(
        default=None,
        description="Generated report content",
    )
    report_format: str | None = Field(
        default=None,
        description="Report format (markdown, plain)",
    )

    # Review results
    review_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Review score (0.0-1.0)",
    )
    review_approved: bool | None = Field(
        default=None,
        description="Whether report was approved",
    )
    review_suggestions: list[Any] | None = Field(
        default=None,
        description="Reviewer suggestions (can be strings or dicts)",
    )
    review_iterations: int | None = Field(
        default=None,
        description="Number of review iterations",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "123e4567-e89b-12d3-a456-426614174000",
                    "status": "completed",
                    "topic": "What are the environmental impacts of electric vehicles?",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:45:00Z",
                    "sources": [
                        {"title": "EPA Study on EVs", "url": "https://epa.gov/ev-study"}
                    ],
                    "findings": ["EVs reduce carbon emissions by 60%"],
                    "claims_verified": 5,
                    "claims_partially_verified": 2,
                    "claims_disputed": 0,
                    "claims_unverified": 1,
                    "insights": ["Battery production has environmental costs"],
                    "report_title": "Environmental Impact of Electric Vehicles",
                    "report_content": "# Environmental Impact...",
                    "report_format": "markdown",
                    "review_score": 0.85,
                    "review_approved": True,
                    "review_suggestions": [],
                    "review_iterations": 2,
                }
            ]
        }
    }
