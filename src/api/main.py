"""FastAPI main application for Veritas API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as research_router
from src.infrastructure.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    setup_logging()
    logger.info("Veritas API starting up...")
    yield
    # Shutdown
    logger.info("Veritas API shutting down...")


app = FastAPI(
    title="Veritas API",
    description="Autonomous Research & Report Generation Platform - API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(research_router)


@app.get("/api/v1/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "veritas-api"}
