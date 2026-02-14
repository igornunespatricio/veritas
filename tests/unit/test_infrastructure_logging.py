"""Unit tests for logging infrastructure."""

from unittest.mock import MagicMock, patch


class TestLoggingSetup:
    """Tests for logging setup functions."""

    @patch("src.infrastructure.logging.logger")
    def test_setup_logging_development(self, mock_logger):
        """Test logging setup in development mode."""
        from src.infrastructure.logging import setup_logging

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.environment = "development"

        with patch("src.infrastructure.logging.settings", mock_settings):
            with patch("src.infrastructure.logging.logger.remove"):
                with patch("src.infrastructure.logging.logger.add"):
                    setup_logging()

                    # Should add stderr handler for development
                    mock_logger.add.assert_called()

    @patch("src.infrastructure.logging.logger")
    @patch("src.infrastructure.logging.Path")
    def test_setup_logging_production(self, mock_path, mock_logger):
        """Test logging setup in production mode."""
        from src.infrastructure.logging import setup_logging

        mock_settings = MagicMock()
        mock_settings.environment = "production"

        # Mock path operations
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.mkdir = MagicMock()

        with patch("src.infrastructure.logging.settings", mock_settings):
            with patch("src.infrastructure.logging.logger.remove"):
                with patch("src.infrastructure.logging.logger.add"):
                    setup_logging()

                    # Should add file handler for production
                    mock_logger.add.assert_called()

    @patch("src.infrastructure.logging.logger")
    def test_get_logger(self, mock_logger):
        """Test get_logger function."""
        mock_logger.bind.return_value = MagicMock()

        from src.infrastructure.logging import get_logger

        logger = get_logger(__name__)

        mock_logger.bind.assert_called_once_with(name=__name__)


class TestLogStage:
    """Tests for log_stage function."""

    @patch("src.infrastructure.logging.logger")
    def test_log_stage_with_logger(self, mock_logger):
        """Test log_stage with explicit logger."""
        from src.infrastructure.logging import log_stage

        mock_instance = MagicMock()
        mock_logger.info = mock_instance

        log_stage("RESEARCH", "Gathering data...", mock_instance)

        mock_instance.info.assert_called_once_with("[RESEARCH] Gathering data...")

    def test_log_stage_imports(self):
        """Test that log_stage can be imported."""
        from src.infrastructure.logging import log_stage

        # Verify the function exists and is callable
        assert callable(log_stage)
