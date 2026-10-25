---
name: ray-jobs
description: >
  Start, submit, parametrize, monitor, and troubleshoot Ray jobs on the local
  chesspos cluster (head node 192.168.1.250, 16 CPUs, 31GB RAM). Use when
  running dataset pipeline jobs (generate_hf_dataset), starting or restarting
  the Ray cluster, submitting ray job submit commands, or debugging failed
  jobs (disk space, OOM, dashboard unreachable, missing configs).
---

# Ray jobs on the chesspos cluster

Practical workflow for the persistent single-node cluster on this machine
(head node `192.168.1.250`, 16 CPUs, 31GB RAM, 467GB disk shared with /tmp).
Always invoke Ray through the project venv: `uv run ray ...` (plain `ray` may
not be on PATH).

## 1. Cluster lifecycle

### Check whether the cluster is alive

```bash
uv run ray status              # no --address needed; uses local GCS
pgrep -f gcs_server            # empty means cluster is down
```

Note: `ray status` rejects `--address http://...`; job subcommands
(`ray job ...`) DO accept `--address http://192.168.1.250:8265`.
If a job failed unexpectedly, also check `uptime` — a reboot kills the
cluster and wipes /tmp (no systemd unit exists; the cluster must be
restarted manually).

### Start the head node (conservative memory settings)

```bash
uv run ray stop                 # if remnants exist
sleep 5
RAY_memory_usage_threshold=0.90 \
uv run ray start --head \
    --num-cpus 16 \
    --dashboard-host 0.0.0.0 \
    --object-store-memory 4000000000
```

- `--dashboard-host 0.0.0.0` is REQUIRED. The default binds the dashboard to
  `127.0.0.1` only, and then `ray job submit --address http://192.168.1.250:8265`
  fails with "Connection refused". Verify with
  `ss -tlnp | grep 8265` → should show `0.0.0.0:8265`.
- `--object-store-memory 4000000000` caps the plasma store (Ray's auto value
  can be 7.7GB+). Then check `uv run ray status` shows ~3.73GiB object store.
- `RAY_memory_usage_threshold=0.90` makes Ray kill workers at 90% node memory
  instead of 95%, protecting the machine (31GB RAM, desktop processes share it).

### Stop the cluster

```bash
uv run ray stop
```

## 2. Parametrizing a job

Dataset pipeline jobs are configured by YAML files in `configs/` (gitignored
directory — copy an existing one, e.g. `pipeline_remote_2023_2024.yaml`).
Key fields:

```yaml
repo_name: "patrickfrank1/chess-positions"
batch_size: 100000            # positions per parquet batch
train_ratio: 0.95
dry_run: false                # false = push to HF Hub
resume: true                  # ALWAYS keep true: skips archives already
                              # completed on the Hub (per-archive _COMPLETE
                              # markers), so restarts are cheap
create_card: false
archive_urls:                 # lichess_db_standard_rated_YYYY-MM.pgn.zst
  - https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst
chunk_target_mb: 64           # PGN chunk size; smaller = smaller Ray task
                              # results = lower peak disk/memory
preprocessing:
  worker_count: 2             # concurrent extraction tasks; each holds ~2GB
                              # in RAM plus its in-flight chunk results
  memory_limit_mb: 2048
  debug: false
sampling:
  tiers: [...]                # Elo-based game subsampling
  min_time_control_seconds: 300
```

Conservative defaults for unattended runs on this machine: `worker_count: 2`,
`chunk_target_mb: 64`. A monthly 2023/2024 lichess archive is ~30GB
compressed; at these settings an archive takes roughly 1-2 hours.

## 3. Submitting a job

```bash
RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 \
uv run ray job submit \
    --address http://192.168.1.250:8265 \
    --no-wait \
    --runtime-env-json '{"working_dir": ".", "excludes": [".git", "data", ".venv", ".ruff_cache", ".pytest_cache", "mlruns"]}' \
    -- /home/pafrank/coding/chesspos/.venv/bin/python3 \
       -m src.run.generate_hf_dataset \
       --config configs/<config>.yaml
```

- `RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1` is CRITICAL: Ray packaging respects
  `.gitignore`, and `configs/.gitignore` ignores every config file. Without
  this the job fails immediately with `Config file not found`.
- The `excludes` list keeps the uploaded zip small (`.git` alone is 48MB,
  `data/` is 6.9GB).
- `--no-wait` returns the job id immediately; the job outlives the shell.
- The uploaded working-dir snapshot is the code AS ON DISK (uncommitted
  changes included). Verify before submitting.

After submitting, confirm the config was picked up:

```bash
uv run ray job logs <job_id> --address http://192.168.1.250:8265 | head -15
# Expect: "Processing N archive(s)", chunk target, worker count, tiers
```

## 4. Monitoring a job

### 4a. Status and logs (CLI)

```bash
# One-shot status: RUNNING / SUCCEEDED / FAILED / STOPPED
uv run ray job status <job_id> --address http://192.168.1.250:8265

# Last lines of driver output (progress bars, errors, tracebacks)
uv run ray job logs <job_id> --address http://192.168.1.250:8265 | tail -30

# Stream live (like tail -f); Ctrl-C detaches WITHOUT killing the job
uv run ray job logs <job_id> --address http://192.168.1.250:8265 --follow

# Find job ids of recent runs (newest last)
uv run ray job list --address http://192.168.1.250:8265

# Cancel a running job
uv run ray job stop <job_id> --address http://192.168.1.250:8265
```

`ray job logs` shows the DRIVER's stdout/stderr only. Worker (extraction
task) errors surface in the driver output only when they fail the job;
for per-worker detail use the dashboard (§4b) or the session log files
(§4c).

What healthy dataset-pipeline logs look like, in order:

1. Startup: `Processing N archive(s) for patrickfrank1/chess-positions`,
   `Chunk target: ...MB`, `Workers: N`, subsampling tiers, time-control
   filter, then `Connected to Ray cluster`.
2. Per archive: HF upload progress bars like
   `...2024-03/batch_0000.parquet: 100%|...| 14.1MB / 14.1MB` and
   `New Data Upload : 100%|...|` — each batch parquet uploads in seconds
   to a couple of minutes.
3. Long SILENT gaps are NORMAL: the slow part is extracting a ~30GB
   archive between batch uploads. Don't restart a "quiet" job.
4. A crash ends with a Python traceback (`OSError: [Errno 28] No space
   left on device`, or a Ray `Task was killed due to the node running
   low on memory` report listing top memory users — that report doubles
   as a diagnosis of WHICH process ate the RAM).

Quick health checks without reading the whole log:

```bash
uv run ray job logs <job_id> --address http://192.168.1.250:8265 \
  | grep -oE "20[0-9]{2}-[0-9]{2}/batch_[0-9]+" | tail -3   # latest batches
uv run ray job status <job_id> --address http://192.168.1.250:8265 | grep -i status
df -h / | tail -1                                          # disk during run
```

### 4b. Dashboard (browser)

Open `http://192.168.1.250:8265` (only reachable if the head was started
with `--dashboard-host 0.0.0.0`, see §1).

- **Jobs** tab: all submitted jobs with status, entrypoint, and — click a
  job — the full driver log with live tail. The log box here matches
  `ray job logs` output.
- **Cluster** tab: node CPU/RAM/GPU utilization, and the resource breakdown
  (used vs available CPUs, memory, object store). Watch `Memory` here while
  a job runs: extraction workers each show ~2GB, the driver 1-8GB.
- **Metrics** view: time-series of node memory, CPU, and per-task
  resources — useful to correlate an OOM/disk crash with what the job was
  doing at that timestamp.
- **Jobs → <job> → Logs** also links to worker log files; task-level
  failures (e.g. one killed `_extract_positions`) show their stderr there.

### 4c. Log files on disk (deep debugging)

Driver and session logs live under the current session directory:

```bash
ls -lh /tmp/ray/session_latest/logs/job-driver-*.log   # driver output (same as ray job logs)
ls -lh /tmp/ray/session_latest/logs/python-core-*.log  # ray core (gcs, raylet, workers)
grep -h "file_system_monitor" /tmp/ray/session_latest/logs/raylet.out | tail -5
    # disk-pressure warnings with timestamps + free space
grep -h -E "Spill|spilled" /tmp/ray/session_latest/logs/raylet.out | tail -5
    # object-store spill activity (0 cumulative = plasma never overflowed)
```

Note `/tmp/ray/session_latest` is a symlink to the active session; after a
reboot it is wiped, and dead sessions remain as
`/tmp/ray/session_<date>_<time>_*` (they can be cleaned when idle, see §5).

### 4d. Disk watchdog for unattended runs

Start before a long run; it snapshots df, temp dir sizes, and top memory
processes every 5 minutes:

```bash
setsid /tmp/opencode/disk_watch.sh >/dev/null 2>&1 < /dev/null &
```

Read it when diagnosing a crash (`tail -40 ~/chesspos_disk_watch.log`) —
the timestamped entries show exactly which path was growing and which
process was heavy when the disk or RAM filled. If the script is missing
(recreates nothing after reboot), rewrite it per §5 of this skill or ask
the operator; its loop appends to `~/chesspos_disk_watch.log`:

```bash
# /tmp/opencode/disk_watch.sh
#!/bin/bash
LOG="$HOME/chesspos_disk_watch.log"
echo "=== watchdog started $(date) ===" >> "$LOG"
while true; do
    {
        echo "--- $(date)"
        df -h / | tail -1
        du -s --block-size=1M /tmp/chesspos-* /tmp/ray 2>/dev/null
        ps aux --sort=-rss | awk 'NR<=4 && $2 != NR {printf "%s %s%% %s %s\n", $2, $4, $11, $12}'
    } >> "$LOG" 2>&1
    sleep 300
done
```

## 5. Troubleshooting playbook

Symptom → cause → fix:

- **`Config file not found: configs/...yaml`** — gitignored file not packaged.
  Resubmit with `RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1` (see above).
- **`Connection refused` on port 8265** — dashboard bound to 127.0.0.1 (head
  started without `--dashboard-host 0.0.0.0`), or cluster down. Check
  `ss -tlnp | grep 8265`, restart head accordingly.
- **`ValueError: Invalid address format: http://...`** from `ray status` —
  `ray status` does not take dashboard URLs. Use plain `uv run ray status`
  or the `RAY_ADDRESS` env var.
- **`OSError: [Errno 28] No space left on device`** — disk full. Inspect
  `/tmp/chesspos-prefetch-*` (should hold ≤2 archive files ≈60GB) and
  `/tmp/ray`. After a job dies, its tempdirs are cleaned automatically, but
  STALE dirs from killed jobs can remain: `rm -rf /tmp/chesspos-*` is safe
  when no job is running. Then resubmit — `resume: true` skips completed
  archives.
- **`Task was killed due to the node running low on memory`** — node OOM
  (30GB/31GB). Check `ps aux --sort=-rss` for non-Ray hogs. Restart cluster
  with the conservative settings in §1 and reduce `worker_count` /
  `chunk_target_mb` in the config. The job dies but the cluster usually
  survives; resume makes restart cheap.
- **Cluster gone entirely after a failure** — check `uptime` for a reboot.
  Restart the head node (§1) and resubmit.
- **Stale Ray sessions filling /tmp** — each dead session leaves ~100MB in
  `/tmp/ray/session_*`. Remove all but the newest when the cluster is idle.

## 6. Rules

- Never read or print environment variable values (secrets such as HF_TOKEN
  live there); referencing names is fine.
- `resume: true` in every real run — it makes any crash recoverable.
- Only ONE dataset job at a time; each run lists and skips `_COMPLETE`
  archives on the Hub, so concurrent jobs would duplicate work.
- After any crash: `df -h /`, clean `/tmp/chesspos-*` only if no job is
  running, then resubmit.
