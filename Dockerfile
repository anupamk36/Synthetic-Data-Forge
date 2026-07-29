FROM python:3.11-slim

LABEL maintainer="Synthetic-Data-Forge Team"

# Security: run as non-root user
RUN groupadd -r forge && useradd -r -g forge -m forge

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ core/
COPY api/ api/
COPY demo_data/ demo_data/

# Create output directory writable by forge user
RUN mkdir -p /app/output_data && chown -R forge:forge /app

USER forge

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8100/health', timeout=5).raise_for_status()" || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "api.server:app", \
    "--host", "0.0.0.0", \
    "--port", "8100"]
