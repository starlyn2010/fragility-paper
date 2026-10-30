FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only torch keeps the image small; the package metadata lives in pyproject.toml.
RUN python -m pip install --upgrade pip \
 && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY data ./data

RUN python -m pip install -e ".[dev]"

# The fast CPU suite; full experiments need the checkpoints under runs/.
CMD ["python", "-m", "pytest", "tests", "-q"]
