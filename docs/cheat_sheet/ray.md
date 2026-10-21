# Ray CLI Cheat Sheet

Practical commands for managing Ray clusters and jobs in this repo.
Ray is used by `src/dataset/etl.py` and `src/dataset/archive_pipeline.py`
(local instance by default, cluster via `RAY_ADDRESS`), and training is
submitted with `ray job submit` (see `src/run/train_ray.py`).

## Targeting a cluster

Most commands act on "the current cluster", which is either a local
instance or whatever `RAY_ADDRESS` points at.

```bash
# Point the CLI at a remote cluster (head node dashboard port)
export RAY_ADDRESS=http://<head-node>:8265

# Or pass the address per command
ray status --address http://<head-node>:8265
ray job list --address http://<head-node>:8265
```

## Cluster status

```bash
ray status                 # CPUs/GPUs, memory, object store, pending tasks
ray list nodes             # nodes in the cluster
ray list actors            # live actors
ray list tasks --limit 20  # recent tasks
ray list jobs              # all jobs the cluster has seen
```

## Jobs

```bash
# Submit a job (from a directory with your code/requirements)
ray job submit --address http://<head-node>:8265 \
    --working-dir . \
    -- python src/run/train_ray.py

ray job list                     # list submitted jobs
ray job status <job_id>          # status of one job
ray job logs <job_id>            # logs of one job
ray job logs --follow <job_id>   # stream logs live
ray job stop <job_id>            # cancel a running job
```

Job IDs look like `raysubmit_XXXX`. `ray job list --output json` is handy
for scripting/filtering.

## Local instance

```bash
ray start --head --num-cpus 8   # start a local head node manually
ray stop                        # stop all local Ray processes
ray status                      # inspect the local instance
```

Note: `ray.init()` in this repo starts an ephemeral local instance on its
own, so manual `ray start` is only needed if you want a persistent one.

## Monitoring UI

- Dashboard: `http://<head-node>:8265` (jobs, actors, tasks, memory)
- Metrics UI is part of the dashboard; logs are browsable per job.

## Environment variables

| Variable      | Purpose                                        |
| ------------- | ---------------------------------------------- |
| `RAY_ADDRESS` | Cluster the CLI / `ray.init()` connects to     |
| `RAY_DEDUP_LOGS=0` | Disable worker log deduplication          |
| `RAY_memory_monitor_refresh_ms` | Tune OOM monitor interval    |

## Troubleshooting quickies

```bash
ray status                      # is the cluster reachable at all?
ray job logs <job_id> | tail    # last output of a failed job
ray list tasks --filter 'state=FAILED' --limit 50
```

- "Connection refused" → wrong host/port, or head node down.
- OOM on workers → reduce `object_store_memory` / batch sizes; check the
  dashboard memory view.
- Local leftovers after a crashed run → `ray stop`.

## See also

- `docs/dataset_pipeline.md` — how Ray Data is used in the ETL.
- `src/run/train_ray.py` — training job entry point.
