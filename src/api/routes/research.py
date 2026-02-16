"""Research endpoints for the Veritas API."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.models.request import ResearchRequest
from src.api.models.response import (
    JobStatus,
    ResearchJobResponse,
    ResearchSource,
    ResearchStatusResponse,
)
from src.orchestration import ResearchWorkflow, WorkflowResult, WorkflowStage

router = APIRouter(prefix="/api/v1/research", tags=["research"])

# In-memory job storage (use Redis/database for production)
_jobs: dict[str, dict[str, Any]] = {}


def _map_workflow_stage_to_progress(stage: WorkflowStage) -> tuple[str, int]:
    """Map workflow stage to progress percentage."""
    stage_map = {
        WorkflowStage.RESEARCH: ("research", 20),
        WorkflowStage.FACT_CHECK: ("fact_check", 40),
        WorkflowStage.SYNTHESIS: ("synthesis", 60),
        WorkflowStage.WRITING: ("writing", 80),
        WorkflowStage.REVIEW: ("review", 90),
        WorkflowStage.COMPLETED: ("completed", 100),
        WorkflowStage.FAILED: ("failed", 100),
    }
    return stage_map.get(stage, ("unknown", 0))


def _convert_workflow_result(
    job_id: str, result: WorkflowResult
) -> ResearchJobResponse:
    """Convert WorkflowResult to API response model."""
    now = datetime.now(UTC)
    created_at = job_id in _jobs and _jobs[job_id].get("created_at") or now

    # Get stage and progress
    current_stage, progress_percentage = _map_workflow_stage_to_progress(result.status)

    # Build base response
    response_data: dict[str, Any] = {
        "job_id": job_id,
        "status": (
            JobStatus.COMPLETED
            if result.status == WorkflowStage.COMPLETED
            else JobStatus.FAILED
        ),
        "topic": "",
        "created_at": created_at,
        "updated_at": now,
        "current_stage": current_stage,
        "progress_percentage": progress_percentage,
    }

    if result.status == WorkflowStage.FAILED:
        response_data["status"] = JobStatus.FAILED
        response_data["error"] = result.error
        return ResearchJobResponse(**response_data)

    # Add research results
    if result.research:
        response_data["topic"] = result.research.topic
        response_data["sources"] = [
            ResearchSource(title=s.get("title", ""), url=s.get("url", ""))
            for s in result.research.sources
        ]
        response_data["findings"] = result.research.findings

    # Add fact-check results
    if result.fact_check:
        claims = result.fact_check.claims
        response_data["claims_verified"] = len(
            [c for c in claims if c.get("status") == "verified"]
        )
        response_data["claims_partially_verified"] = len(
            [c for c in claims if c.get("status") == "partially_verified"]
        )
        response_data["claims_disputed"] = len(
            [c for c in claims if c.get("status") == "disputed"]
        )
        response_data["claims_unverified"] = len(
            [c for c in claims if c.get("status") == "unverified"]
        )

    # Add synthesis results
    if result.synthesis:
        response_data["insights"] = result.synthesis.insights

    # Add report results
    if result.report:
        response_data["report_title"] = result.report.title
        response_data["report_content"] = result.report.content
        response_data["report_format"] = result.report.format

    # Add review results
    if result.review:
        response_data["review_score"] = result.review.score
        response_data["review_approved"] = result.review.approved
        response_data["review_suggestions"] = result.review.suggestions
        response_data["review_iterations"] = result.iterations

    return ResearchJobResponse(**response_data)


async def _run_research_workflow(job_id: str, request: ResearchRequest) -> None:
    """Background task to run the research workflow."""
    try:
        # Update status to processing
        _jobs[job_id]["status"] = JobStatus.PROCESSING
        _jobs[job_id]["current_stage"] = "research"
        _jobs[job_id]["progress_percentage"] = 20

        # Create and execute workflow
        workflow = ResearchWorkflow(
            max_iterations=request.max_iterations,
            auto_approve_threshold=request.auto_approve_threshold,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            max_tokens=request.max_tokens,
        )

        result = await workflow.execute(request.topic, correlation_id=job_id)

        # Store result
        _jobs[job_id]["result"] = result
        _jobs[job_id]["status"] = (
            JobStatus.COMPLETED
            if result.status == WorkflowStage.COMPLETED
            else JobStatus.FAILED
        )
        _jobs[job_id]["current_stage"] = result.status.value
        _jobs[job_id]["progress_percentage"] = 100
        _jobs[job_id]["updated_at"] = datetime.now(UTC)

    except Exception as e:
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["updated_at"] = datetime.now(UTC)


@router.post(
    "",
    response_model=ResearchStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a research job",
    description="Submit a new research topic for analysis. Returns a job ID for tracking.",
)
async def submit_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchStatusResponse:
    """Submit a new research job to be processed in the background."""
    job_id = str(uuid4())
    now = datetime.now(UTC)

    # Store job info
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "topic": request.topic,
        "created_at": now,
        "updated_at": now,
        "current_stage": None,
        "progress_percentage": 0,
        "request": request.model_dump(),
    }

    # Add background task
    background_tasks.add_task(_run_research_workflow, job_id, request)

    return ResearchStatusResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        topic=request.topic,
        created_at=now,
        updated_at=now,
    )


@router.get(
    "/{job_id}",
    response_model=ResearchJobResponse,
    summary="Get research job status and results",
    description="Retrieve the current status and results of a research job.",
)
async def get_research_job(job_id: str) -> ResearchJobResponse:
    """Get the status and results of a research job."""
    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    job = _jobs[job_id]
    now = datetime.now(UTC)

    # If job is still pending/processing, return status only
    if job["status"] in (JobStatus.PENDING, JobStatus.PROCESSING):
        return ResearchJobResponse(
            job_id=job_id,
            status=job["status"],
            topic=job["topic"],
            created_at=job["created_at"],
            updated_at=job.get("updated_at", now),
            current_stage=job.get("current_stage"),
            progress_percentage=job.get("progress_percentage"),
        )

    # If completed/failed, return full results
    if "result" in job:
        return _convert_workflow_result(job_id, job["result"])

    # Failed job without result
    return ResearchJobResponse(
        job_id=job_id,
        status=job["status"],
        topic=job["topic"],
        created_at=job["created_at"],
        updated_at=job.get("updated_at", now),
        error=job.get("error"),
    )


@router.get(
    "",
    response_model=list[ResearchStatusResponse],
    summary="List all research jobs",
    description="Get a list of all research jobs and their statuses.",
)
async def list_research_jobs() -> list[ResearchStatusResponse]:
    """List all research jobs."""
    jobs = []
    for job_id, job in _jobs.items():
        if job["status"] in (JobStatus.PENDING, JobStatus.PROCESSING):
            jobs.append(
                ResearchStatusResponse(
                    job_id=job_id,
                    status=job["status"],
                    topic=job["topic"],
                    created_at=job["created_at"],
                    updated_at=job.get("updated_at", datetime.now(UTC)),
                    current_stage=job.get("current_stage"),
                    progress_percentage=job.get("progress_percentage"),
                )
            )
    return jobs


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a research job",
    description="Remove a job from the tracking system.",
)
async def delete_research_job(job_id: str) -> None:
    """Delete a research job."""
    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    del _jobs[job_id]
