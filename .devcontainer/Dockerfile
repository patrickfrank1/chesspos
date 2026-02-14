FROM python:3.11-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    wget \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install opencode
RUN curl -fsSL https://opencode.ai/install | bash

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | bash

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install ruff and ty via uv
RUN uv tool install ruff \
    && uv tool install ty

# Set working directory
WORKDIR /workspace
