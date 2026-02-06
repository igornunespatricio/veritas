# Veritas

Autonomous Research & Report Generation Platform

A production-grade multi-AI-agent system that collaboratively produces high-quality research reports on arbitrary topics.

## Architecture

```
src/
├── domain/          # Business entities, interfaces, and core logic
├── agents/          # Agent implementations (Researcher, Fact-Checker, Synthesizer, Writer, Critic)
├── orchestration/   # Workflow orchestration and agent coordination
├── infrastructure/  # LLM clients, API wrappers, and external services
└── config/          # Settings and configuration management
tests/
├── unit/            # Unit tests for agents and domain logic
└── integration/     # Integration tests for workflows
```

## Agents

1. **Researcher Agent** - Collects raw information, sources, and key findings
2. **Fact-Checker Agent** - Verifies claims and assigns confidence scores
3. **Synthesizer Agent** - Merges validated research into coherent insights
4. **Writer Agent** - Produces polished, structured reports
5. **Critic Agent** - Reviews reports for clarity, logic, and completeness

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
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
LANGSMITH_API_KEY=optional-langsmith-key
```

## License

MIT
