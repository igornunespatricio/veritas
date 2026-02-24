# Veritas

Autonomous Research & Report Generation Platform

A production-grade multi-AI-agent system that collaboratively produces high-quality research reports on arbitrary topics. Built with LangChain for LLM orchestration, FastAPI for the REST API, and designed for production deployment with resilience patterns.

## Architecture

```
src/
├── api/                  # FastAPI REST API
│   ├── main.py          # Application entry point
│   ├── models/         # Pydantic request/response models
│   └── routes/        # API endpoints (research, health)
├── agents/              # Agent implementations
│   ├── base.py        # Base agent class with LangChain
│   ├── researcher.py  # Researcher agent with web search
│   ├── factchecker.py # Fact-checker agent
│   ├── synthesizer.py # Synthesizer agent
│   ├── writer.py      # Writer agent
│   └── critic.py      # Critic agent
├── domain/              # Business entities and interfaces
│   ├── interfaces.py  # Agent contracts and registry
│   └── events.py      # Domain events for agent communication
├── infrastructure/      # External services and utilities
│   ├── llm.py         # Multi-provider LLM clients (OpenAI, Anthropic, Ollama)
│   ├── tools.py       # Web search tool (Tavily)
│   ├── circuit_breaker.py # Circuit breaker pattern
│   └── logging.py     # Structured logging with correlation IDs
├── orchestration/       # Workflow orchestration
│   └── workflow.py    # Research workflow coordinator
└── config/             # Settings and configuration
    ├── __init__.py   # Environment-based settings
    └── retry.py      # Retry configuration
tests/
├── unit/               # Unit tests for agents and domain logic
└── integration/        # Integration tests for workflows and API
```

## Agents

1. **Researcher Agent** - Collects raw information, sources, and key findings using web search (Tavily)
2. **Fact-Checker Agent** - Verifies claims and assigns confidence scores
3. **Synthesizer Agent** - Merges validated research into coherent insights
4. **Writer Agent** - Produces polished, structured reports in Markdown or plain text
5. **Critic Agent** - Reviews reports for clarity, logic, and completeness with scoring

### Agent Communication

Agents communicate through explicit **Domain Events**:

- `ResearchCompleted` - Contains topic, sources, and findings
- `FactCheckCompleted` - Contains verified claims and confidence scores
- `SynthesisCompleted` - Contains merged insights and resolved contradictions
- `ReportWritten` - Contains title, content, and format
- `ReportReviewed` - Contains suggestions, score, and approval status

## Workflow

The system supports two workflow modes:

### Sequential Workflow

Single-pass execution: Researcher → Fact-Checker → Synthesizer → Writer

### Iterative Workflow (Default)

With review iterations:

```
Researcher → Fact-Checker → Synthesizer → Writer → Critic
                                              ↓
                                    (revision if needed)
                                              ↓
                                            Writer → Critic → ...
```

Configure via `ResearchWorkflow` parameters:

- `max_iterations`: Maximum review iterations (default: 3)
- `auto_approve_threshold`: Score threshold for auto-approval (default: 0.8)

## API

Veritas provides a RESTful API built with FastAPI:

### Endpoints

| Method | Endpoint                    | Description                |
| ------ | --------------------------- | -------------------------- |
| POST   | `/api/v1/research`          | Submit a new research job  |
| GET    | `/api/v1/research/{job_id}` | Get job status and results |
| GET    | `/api/v1/research`          | List jobs with filtering   |
| DELETE | `/api/v1/research/{job_id}` | Delete a job               |
| GET    | `/api/v1/health`            | Health check               |

### Request Example

```bash
curl -X POST "http://localhost:8000/api/v1/research" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The impact of AI on software development",
    "max_iterations": 3,
    "llm_provider": "openai",
    "llm_model": "gpt-4o"
  }'
```

### Response Example

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "topic": "The impact of AI on software development",
  "sources": [
    {
      "title": "AI in Software Development",
      "url": "https://example.com/ai-dev"
    }
  ],
  "findings": ["Finding 1", "Finding 2"],
  "report_title": "The Impact of AI on Software Development",
  "report_content": "# The Impact of AI on Software Development\n\n...",
  "review_score": 0.85,
  "review_approved": true,
  "review_iterations": 2
}
```

## LLM Providers

The system supports multiple LLM providers through LangChain:

| Provider       | Default Model              | Environment Variable |
| -------------- | -------------------------- | -------------------- |
| OpenAI         | `gpt-4o`                   | `OPENAI_API_KEY`     |
| Anthropic      | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY`  |
| OpenRouter     | `openai/gpt-5-nano`        | `OPENROUTER_API_KEY` |
| Ollama (local) | `llama3.2:3b`              | `OLLAMA_BASE_URL`    |

## Resilience Features

### Retry Configuration

Automatic retry with exponential backoff for transient failures:

- Configurable max attempts
- Exponential backoff multiplier
- Min/max delay bounds
- Configured via environment or `RetryConfig`

### Circuit Breaker

Prevents cascading failures with three states:

- **Closed**: Normal operation
- **Open**: Failing, reject all requests
- **Half-Open**: Testing if service recovered

Configure via:

- `CIRCUIT_FAILURE_THRESHOLD`: Failures before opening (default: 5)
- `CIRCUIT_COOLDOWN_SECONDS`: Cooldown period (default: 30)
- `CIRCUIT_TIMEOUT_SECONDS`: Per-call timeout (default: 60)

### Correlation IDs

Every request is traced with correlation IDs for observability.

## Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests
pytest

# Lint and format
ruff check .
ruff format .
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# LLM Providers
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENROUTER_API_KEY=your-openrouter-key

# Local LLM (Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Web Search
TAVILY_API_KEY=your-tavily-key

# Observability (Optional)
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=false

# Environment
ENVIRONMENT=development

# Retry Configuration
RETRY_MAX_ATTEMPTS=5
RETRY_MAX_BACKOFF=60.0
RETRY_BASE_DELAY=2.0

# Circuit Breaker Configuration
CIRCUIT_FAILURE_THRESHOLD=5
CIRCUIT_COOLDOWN_SECONDS=30.0
CIRCUIT_TIMEOUT_SECONDS=60.0
```

## Deployment

### Docker

```bash
# Build the image
docker build -t veritas .

# Run the container
docker run -p 8000:8000 --env-file .env veritas
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
```

The API will be available at `http://localhost:8000` with interactive docs at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v
```

### Test Coverage

- **Unit Tests**: Agents, domain logic, infrastructure (LLM, logging, tools, circuit breaker), config
- **Integration Tests**: Multi-agent workflows, API endpoints, error handling, end-to-end scenarios

## Development

```bash
# Run the API locally
uvicorn src.api.main:app --reload

# Run with custom settings
uvicorn src.api.main:app --reload --env-file .env

# Access interactive API docs
open http://localhost:8000/docs
```

## License

MIT
