FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./
COPY src/ ./src/

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
