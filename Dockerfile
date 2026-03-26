FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync

# Copy project files
COPY prepare.py train.py sample.py ./
COPY program.md ./

# Cache dir for data — mount a volume here to persist across runs
ENV AUTORESEARCH_CACHE="/root/.cache/autoresearch"

# Default: prepare data then run the before/after comparison
CMD ["sh", "-c", "uv run prepare.py --num-shards 30 && uv run sample.py"]
