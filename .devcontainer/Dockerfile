FROM python:3.11.8-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.13.6
ARG OPENCODE_VERSION=1.19.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    gnupg \
    jq \
    less \
    libsecret-1-0 \
    libx11-6 \
    libxkbfile1 \
    openssh-client \
    procps \
    ripgrep \
    unzip \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_VERSION="$UV_VERSION" UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

RUN curl -fsSL https://opencode.ai/install \
    | bash -s -- --version "$OPENCODE_VERSION" \
    && install -m 0755 /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && rm -rf /root/.opencode

RUN groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid dev --create-home --shell /bin/bash dev \
    && mkdir -p /workspace /home/dev/.cache /home/dev/.config \
       /home/dev/.local/share /home/dev/.vscode-server \
    && chown -R dev:dev /workspace /home/dev

ENV PATH="/home/dev/.local/bin:${PATH}" \
    UV_LINK_MODE=copy

USER dev
WORKDIR /workspace
