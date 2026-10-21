FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends make postgresql-client \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["make", "help"]
