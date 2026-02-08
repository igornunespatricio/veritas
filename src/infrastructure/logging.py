"""Logging configuration for Veritas."""

import sys
from pathlib import Path
from loguru import logger

from src.config import settings


def setup_logging():
    """Configure logging based on environment.

    Development: Outputs to stdout with colors for visibility
    Production: Outputs to rotating log files for persistence
    """
    # Remove default handler
    logger.remove()

    if settings.environment == "production":
        # Production: Structured logging to file with rotation
        log_path = Path("logs/veritas.log")
        log_path.parent.mkdir(exist_ok=True)

        logger.add(
            str(log_path),
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            enqueue=True,  # Non-blocking for production
        )
    else:
        # Development: Simple stdout with colors for visibility
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            colorize=True,
        )


def get_logger(name: str):
    """Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Convenience function for progress output
def log_stage(stage: str, message: str, logger_instance=None):
    """Log a workflow stage with visual formatting.

    Args:
        stage: Current stage name (e.g., "RESEARCH", "FACT-CHECK")
        message: Status message
        logger_instance: Optional logger instance
    """
    logger_obj = logger_instance or logger
    logger_obj.info(f"[{stage}] {message}")
