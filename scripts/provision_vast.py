"""Provision a vast.ai GPU instance for chesspos training.

Reads VASTAI_API_KEY from the environment. The key is never logged or
persisted; it is passed to the vastai CLI per invocation.

Examples:
    python scripts/provision_vast.py
    python scripts/provision_vast.py --gpu RTX_4090 --top 5
    python scripts/provision_vast.py --instance-id 12345
    python scripts/provision_vast.py --destroy 12345
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision a vast.ai instance for chesspos training"
    )
    parser.add_argument("--gpu", default="RTX_4090")
    parser.add_argument("--disk", type=int, default=40)
    parser.add_argument("--image", default="pytorch/pytorch")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--min-ram", type=int, default=30000, help="MB")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="skip search/confirm/create; just wait for the instance to run",
    )
    parser.add_argument(
        "--destroy", type=int, default=None, help="destroy the given instance"
    )
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    return parser.parse_args()


def vastai(*args: str) -> str:
    result = subprocess.run(
        ["vastai", *args, "--api-key", os.environ["VASTAI_API_KEY"]],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def require_api_key() -> str:
    key = os.environ.get("VASTAI_API_KEY")
    if not key:
        sys.exit(
            "VASTAI_API_KEY is not set. Export it (operator responsibility) "
            "and re-run. Cannot proceed without it."
        )
    return key


def search_offers(args: argparse.Namespace) -> list[dict]:
    query = (
        f"gpu_name={args.gpu} num_gpus=1 verified=true rentable=true "
        "disk_space>=40 cpu_cores>=8 reliability>0.98"
    )
    output = vastai(
        "search", "offers", query, "-o", "dph_total", "--raw", "--limit", "100"
    )
    offers = json.loads(output)
    if isinstance(offers, dict):
        offers = offers.get("offers", [])
    return [
        offer
        for offer in offers
        if (offer.get("cpu_ram") or 0) >= args.min_ram
        and int(str(offer.get("driver_version", "0")).split(".")[0]) >= 580
    ]


def format_offer(offer: dict) -> str:
    def get(*keys: str) -> str:
        for key in keys:
            value = offer.get(key)
            if value:
                return str(value)
        return "?"

    def num(key: str, digits: int = 0) -> str:
        value = offer.get(key)
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "?"

    return (
        f"id={get('id')} "
        f"gpu={get('gpu_name')} "
        f"${num('dph_total', 2)}/h "
        f"cpu={num('cpu_cores')}cores "
        f"ram={num('cpu_ram')}MB "
        f"disk={num('disk_space', 1)}GB "
        f"driver={get('driver_version')} "
        f"net={num('inet_down')}Mbit-down/{num('inet_up')}Mbit-up "
        f"rel={num('reliability2', 4)} "
        f"location={get('geolocation')}"
    )


def confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(f"{message} [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def create_instance(args: argparse.Namespace, offer_id: int) -> int:
    output = vastai(
        "create",
        "instance",
        str(offer_id),
        "--image",
        args.image,
        "--disk",
        str(args.disk),
        "--ssh",
        "--direct",
        "--raw",
    )
    try:
        payload = json.loads(output)
        if not payload.get("success", False):
            sys.exit(
                f"create returned success={payload.get('success')} "
                "(verify no phantom contract exists: vastai show instances)"
            )
        instance_id = int(payload["new_contract"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r"new_(?:contract|instance_id):\s*(\d+)", output)
        if not match:
            sys.exit("could not parse instance id from create output")
        instance_id = int(match.group(1))
    return instance_id


def wait_until_running(instance_id: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_status = "?"
    while time.monotonic() < deadline:
        output = vastai("show", "instance", str(instance_id), "--raw")
        payload = json.loads(output)
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        last_status = str(payload.get("actual_status", payload.get("status", "?")))
        if last_status == "running":
            return
        time.sleep(15)
    sys.exit(
        f"instance {instance_id} not running after {timeout}s (status={last_status})"
    )


def destroy_instance(instance_id: int, assume_yes: bool) -> None:
    if not confirm(f"destroy instance {instance_id}?", assume_yes):
        print("aborted")
        return
    output = vastai("destroy", "instance", str(instance_id), "-y")
    print(output.strip())


def main() -> None:
    args = parse_args()
    require_api_key()

    if args.destroy is not None:
        destroy_instance(args.destroy, args.yes)
        return

    instance_id = args.instance_id
    if instance_id is None:
        offers = search_offers(args)
        if not offers:
            sys.exit("no offers matched the query")
        offers = offers[: args.top]
        print("top offers (sorted by price, driver>=580):")
        for offer in offers:
            print(f"  {format_offer(offer)}")

        cheapest = min(offers, key=lambda offer: offer.get("dph_total") or 1e9)
        best_value = offers[0]
        print(
            f"\ncheapest: id={cheapest['id']} (${cheapest['dph_total']}/h) | "
            f"best value: id={best_value['id']} (${best_value['dph_total']}/h)"
        )

        raw_id = input(
            f"offer id to provision (blank = best value {best_value['id']}): "
        ).strip()
        if not raw_id:
            chosen = best_value
        else:
            offer_id = int(raw_id)
            matches = [o for o in offers if int(o["id"]) == offer_id]
            if not matches:
                sys.exit(f"offer id {offer_id} is not among the listed offers")
            chosen = matches[0]

        if not confirm(
            f"provision offer {chosen['id']} at ${chosen['dph_total']}/h "
            f"on {chosen['gpu_name']} (disk={args.disk}GB, image={args.image})?",
            args.yes,
        ):
            print("aborted")
            return
        instance_id = create_instance(args, int(chosen["id"]))

    print(f"instance {instance_id} created; waiting for running state...")
    wait_until_running(instance_id, args.timeout)
    ssh_url = vastai("ssh-url", str(instance_id)).strip()
    print(f"running. connect with:\n  {ssh_url}")
    print(
        "next: HF_TOKEN=... GITHUB_TOKEN=... LAUNCH_TRAINING=1 "
        "bash scripts/gpu_bootstrap.sh  (on the box)"
    )


if __name__ == "__main__":
    main()
