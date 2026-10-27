#!/usr/bin/env bash
set -euo pipefail

git config --global --add safe.directory /workspace
uv sync --extra cpu
