#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SANDBOX=${SANDBOX:-chesspos}

cd "$ROOT"

case "${1:-}" in
    in)
        openshell sandbox upload "$SANDBOX" . /workspace
        ;;
    out)
        staging=$(mktemp -d)
        trap 'rm -rf "$staging"' EXIT
        openshell sandbox download "$SANDBOX" /workspace "$staging"
        source_dir="$staging/workspace"
        [[ -d "$source_dir" ]] || source_dir="$staging"
        (cd "$source_dir" && tar cf - --exclude=./.venv .) | (cd "$ROOT" && tar xf -)
        ;;
    *)
        printf 'usage: %s {in|out}\n' "$0"
        exit 2
        ;;
esac
