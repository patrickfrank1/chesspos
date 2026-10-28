#!/usr/bin/env bash
set -euo pipefail

CALLER_PWD=$PWD
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SANDBOX=${SANDBOX:-chesspos}
IMAGE=${IMAGE:-chesspos-agent:local}
GITHUB_REPO=${GITHUB_REPO:-patrickfrank1/chesspos}
BIND_MOUNT=${BIND_MOUNT:-1}
SANDBOX_HOME=${SANDBOX_HOME:-/home/dev}
OPENCODE_CONFIG_CONTENT=${OPENCODE_CONFIG_CONTENT:-'{"mcp":{"github":{"enabled":false}}}'}
ENV_FILE=${ENV_FILE:-"$ROOT/.env"}
[[ $ENV_FILE = /* ]] || ENV_FILE="$CALLER_PWD/$ENV_FILE"
GIT_USER_NAME=${GIT_USER_NAME:-$(git config --global --get user.name 2>/dev/null || true)}
GIT_USER_EMAIL=${GIT_USER_EMAIL:-$(git config --global --get user.email 2>/dev/null || true)}

cd "$ROOT"

load_env() {
    local file=$1
    local entry key value mode
    local -a env_keys=(
        DEEPSEEK_API_KEY
        ZAI_API_KEY
        GITHUB_TOKEN
        GH_TOKEN
        HF_TOKEN
        HUGGING_FACE_HUB_TOKEN
        VAST_AI_API_KEY
        VASTAI_API_KEY
        VAST_API_KEY
    )
    local -A already_set=()

    [[ -f $file ]] || return

    printf 'loading credentials from %s\n' "$file" >&2

    if [[ $(realpath "$file") == "$ROOT/"* ]]; then
        if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
            printf 'refusing: %s is tracked by git\n' "$file" >&2
            exit 1
        fi
        if ! git check-ignore -q "$file"; then
            printf 'refusing: %s is inside the repository but is not gitignored\n' "$file" >&2
            exit 1
        fi
    fi

    mode=$(stat -c '%a' "$file" 2>/dev/null || stat -f '%Lp' "$file")
    if [[ $mode != 600 && $mode != 400 ]]; then
        printf 'warning: %s has mode %s; run chmod 600 %s\n' "$file" "$mode" "$file" >&2
    fi

    for key in "${env_keys[@]}"; do
        if [[ -n ${!key+x} ]]; then
            already_set[$key]=1
        fi
    done

    while IFS= read -r -d '' entry; do
        key=${entry%%=*}
        value=${entry#*=}
        if [[ -z ${already_set[$key]:-} ]]; then
            export "$key=$value"
        fi
    done < <(
        set +u
        # shellcheck source=/dev/null
        source "$file"
        for key in "${env_keys[@]}"; do
            if [[ -n ${!key+x} ]]; then
                printf '%s=%s\0' "$key" "${!key}"
            fi
        done
    )
    return
}

load_env "$ENV_FILE"

command -v openshell >/dev/null 2>&1 || {
    printf '%s\n' "openshell is not on PATH; install it from https://github.com/NVIDIA/OpenShell"
    exit 1
}

if ! openshell status >/dev/null 2>&1; then
    printf '%s\n' "OpenShell gateway is not reachable on the selected endpoint." >&2
    printf '%s\n' "Start it with: systemctl --user enable --now openshell-gateway" >&2
    printf '%s\n' "Headless VM: sudo loginctl enable-linger \"\$USER\" (once)" >&2
    exit 1
fi

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

create_env=(--env "OPENCODE_CONFIG_CONTENT=$OPENCODE_CONFIG_CONTENT")
create_args=()
if [[ $BIND_MOUNT == 1 ]]; then
    create_env+=(--env "HOME=$SANDBOX_HOME")
    create_args+=(--driver-config-json "{\"docker\":{\"mounts\":[{\"type\":\"bind\",\"source\":\"$ROOT\",\"target\":\"/workspace\",\"read_only\":false}]}}")
else
    create_args+=(--upload .:/workspace)
fi

if openshell sandbox get "$SANDBOX" >/dev/null 2>&1; then
    if [[ $BIND_MOUNT == 1 ]]; then
        printf 'bind mount is active; skipping repository upload\n' >&2
        if ! openshell sandbox exec "$SANDBOX" --workdir /workspace -- test -d /workspace/.git >/dev/null 2>&1; then
            printf 'warning: %s lacks the bind mount; recreate it with RECREATE=1\n' "$SANDBOX" >&2
        fi
    else
        openshell sandbox upload "$SANDBOX" . /workspace
    fi
    for name in "${provider_names[@]}"; do
        openshell sandbox provider attach "$SANDBOX" "$name" >/dev/null 2>&1 || true
    done
else
    openshell sandbox create \
        --name "$SANDBOX" \
        --from "$IMAGE" \
        --policy .openshell/policy.yaml \
        "${create_env[@]}" \
        "${providers[@]}" \
        "${create_args[@]}" \
        --detach
fi

openshell sandbox exec -n "$SANDBOX" --workdir /workspace -- \
    git config --global --add safe.directory /workspace
if [[ -n $GIT_USER_NAME ]]; then
    openshell sandbox exec -n "$SANDBOX" --workdir /workspace -- \
        git config --global user.name "$GIT_USER_NAME"
fi
if [[ -n $GIT_USER_EMAIL ]]; then
    openshell sandbox exec -n "$SANDBOX" --workdir /workspace -- \
        git config --global user.email "$GIT_USER_EMAIL"
fi
openshell sandbox exec -n "$SANDBOX" --workdir /workspace -- uv sync --extra cpu

if [[ ${NO_EDITOR:-0} != 1 ]]; then
    if [[ -x /snap/bin/code ]]; then
        PATH="/snap/bin:$PATH" openshell sandbox connect "$SANDBOX" --editor vscode
    else
        openshell sandbox connect "$SANDBOX" --editor vscode
    fi
fi
