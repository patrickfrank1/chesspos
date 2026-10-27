#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SANDBOX=${SANDBOX:-chesspos}
IMAGE=${IMAGE:-chesspos-agent:local}
GITHUB_REPO=${GITHUB_REPO:-patrickfrank1/chesspos}

cd "$ROOT"

command -v openshell >/dev/null 2>&1 || {
    printf '%s\n' "openshell is not on PATH; install it from https://github.com/NVIDIA/OpenShell"
    exit 1
}

for profile in openshell/providers/*.yaml; do
    id=$(basename "$profile" .yaml)
    if openshell profile describe "$id" --global >/dev/null 2>&1; then
        openshell profile update "$id" -f "$profile" --global
    else
        openshell profile import -f "$profile" --global
    fi
done

create_provider() {
    local name=$1
    local type=$2
    shift 2

    if openshell provider get "$name" >/dev/null 2>&1; then
        return
    fi
    for variable in "$@"; do
        if [[ -n "${!variable:-}" ]]; then
            openshell provider create --name "$name" --type "$type" --from-existing
            return
        fi
    done
}

create_provider deepseek deepseek DEEPSEEK_API_KEY
create_provider zai zai ZAI_API_KEY
create_provider github github GITHUB_TOKEN
create_provider huggingface huggingface HF_TOKEN HUGGING_FACE_HUB_TOKEN
create_provider vastai vastai VAST_AI_API_KEY VASTAI_API_KEY VAST_API_KEY

providers=()
provider_names=()
for name in deepseek zai github huggingface vastai; do
    if openshell provider get "$name" >/dev/null 2>&1; then
        providers+=(--provider "$name")
        provider_names+=("$name")
    fi
done

mkdir -p .openshell
sed "s|__GITHUB_REPO__|${GITHUB_REPO}|g" openshell/policy.yaml >.openshell/policy.yaml

docker build -t "$IMAGE" -f .devcontainer/Dockerfile .

if [[ ${RECREATE:-0} == 1 ]] && openshell sandbox get "$SANDBOX" >/dev/null 2>&1; then
    openshell sandbox delete "$SANDBOX"
fi

if openshell sandbox get "$SANDBOX" >/dev/null 2>&1; then
    openshell sandbox upload "$SANDBOX" . /workspace
    for name in "${provider_names[@]}"; do
        openshell sandbox provider attach "$SANDBOX" "$name" >/dev/null 2>&1 || true
    done
else
    openshell sandbox create \
        --name "$SANDBOX" \
        --from "$IMAGE" \
        --policy .openshell/policy.yaml \
        "${providers[@]}" \
        --detach \
        --upload .:/workspace
fi

openshell sandbox exec -n "$SANDBOX" --workdir /workspace -- \
    git config --global --add safe.directory /workspace
openshell sandbox exec -n "$SANDBOX" --workdir /workspace -- uv sync --extra cpu

if [[ ${NO_EDITOR:-0} != 1 ]]; then
    openshell sandbox connect "$SANDBOX" --editor vscode
fi
