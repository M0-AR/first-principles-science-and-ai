FROM python:3.12-slim

WORKDIR /lab

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /lab/pyproject.toml
COPY src /lab/src
RUN pip install --no-cache-dir -e ".[torch]" --extra-index-url https://download.pytorch.org/whl/cpu

COPY scripts /lab/scripts
COPY Makefile /lab/Makefile

ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/run_all.py"]
