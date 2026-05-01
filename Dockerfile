FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HIV_API_HOST=0.0.0.0 \
    HIV_API_PORT=8080 \
    PORT=8080 \
    HIV_MODELS_DIR=/app/models \
    HIV_ARTIFACTS_DIR=/app/artifacts \
    TF_USE_LEGACY_KERAS=1 \
    PIP_DEFAULT_TIMEOUT=200 \
    PIP_RETRIES=20

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --timeout 200 --retries 20 -r /tmp/requirements.txt

RUN apt-get purge -y --auto-remove build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY app /app/app
COPY src /app/src
COPY models /app/models
COPY main.py /app/main.py

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/models /app/artifacts \
    && chown -R appuser:appuser /app

# USER appuser   <-- TEMPORAIREMENT COMMENTÉ

EXPOSE 8080

CMD ["python", "-m", "main"]