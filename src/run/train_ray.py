"""Run the masked transformer training job on a Ray cluster.

Data is snapshot-downloaded from a HuggingFace dataset repo inside each
training worker; the ETL stays a separate job on the same cluster.

Local run (joins an existing cluster via RAY_ADDRESS, else starts one):

    python -m src.run.train_ray --data-repo user/chess-positions

Cluster submission:

    ray job submit --address http://<head>:8265 \
        --runtime-env-json '{"working_dir": "."}' \
        -- python src/run/train_ray.py --config configs/training.yaml
"""

from __future__ import annotations

import argparse

from src.training.ray_train import (
    RayTrainingConfig,
    build_run_config,
    build_scaling_config,
    build_trainer,
    load_yaml_overrides,
    resolve_config,
    resolve_data_dirs,
    train_loop_per_worker,
)

__all__ = [
    "RayTrainingConfig",
    "build_run_config",
    "build_scaling_config",
    "build_trainer",
    "load_yaml_overrides",
    "main",
    "parse_args",
    "resolve_config",
    "resolve_data_dirs",
    "train_loop_per_worker",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train masked stream transformer on a Ray cluster"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file with a 'training' section; CLI flags override it",
    )
    parser.add_argument(
        "--data-repo",
        dest="data_repo",
        default=None,
        help="HuggingFace dataset repo id; train/val data is snapshot-downloaded "
        "from it in each worker",
    )
    parser.add_argument("--data-cache", dest="data_cache", default=None)
    parser.add_argument(
        "--data-limit-gb",
        dest="data_limit_gb",
        type=float,
        default=None,
        help="Size budget for the train split download (test split is always "
        "downloaded in full)",
    )
    parser.add_argument("--upload-bucket", dest="upload_bucket", default=None)
    parser.add_argument("--upload-prefix", dest="upload_prefix", default=None)
    parser.add_argument("--mlflow-uri", dest="mlflow_uri", default=None)
    parser.add_argument("--mlflow-experiment", dest="mlflow_experiment", default=None)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
    parser.add_argument("--d-model", dest="d_model", type=int, default=None)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=None)
    parser.add_argument("--n-heads", dest="n_heads", type=int, default=None)
    parser.add_argument("--d-ff", dest="d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=None)
    parser.add_argument("--max-segments", dest="max_segments", type=int, default=None)
    parser.add_argument("--p-random", dest="p_random", type=float, default=None)
    parser.add_argument("--p-board", dest="p_board", type=float, default=None)
    parser.add_argument("--p-span", dest="p_span", type=float, default=None)
    parser.add_argument("--p-rect", dest="p_rect", type=float, default=None)
    parser.add_argument("--p-skip-gram", dest="p_skip_gram", type=float, default=None)
    parser.add_argument(
        "--skip-gram-unmask-rate",
        dest="skip_gram_unmask_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--rect-min-rate", dest="rect_min_rate", type=float, default=None
    )
    parser.add_argument(
        "--rect-max-rate", dest="rect_max_rate", type=float, default=None
    )
    parser.add_argument("--rect-min-side", dest="rect_min_side", type=int, default=None)
    parser.add_argument("--rect-max-side", dest="rect_max_side", type=int, default=None)
    parser.add_argument(
        "--no-rect-sync",
        dest="rect_synchronize",
        action="store_false",
        default=None,
        help="Mask independent rectangles per segment",
    )
    parser.add_argument("--mlm-weight", dest="mlm_weight", type=float, default=None)
    parser.add_argument("--inv-weight", dest="inv_weight", type=float, default=None)
    parser.add_argument("--var-weight", dest="var_weight", type=float, default=None)
    parser.add_argument("--cov-weight", dest="cov_weight", type=float, default=None)
    parser.add_argument("--nce-weight", dest="nce_weight", type=float, default=None)
    parser.add_argument(
        "--nce-temperature", dest="nce_temperature", type=float, default=None
    )
    parser.add_argument("--vicreg-gamma", dest="vicreg_gamma", type=float, default=None)
    parser.add_argument("--vicreg-eps", dest="vicreg_eps", type=float, default=None)
    parser.add_argument("--aux-warmup", dest="aux_warmup", type=int, default=None)
    parser.add_argument(
        "--retrieval-benchmark", dest="retrieval_benchmark", default=None
    )
    parser.add_argument(
        "--retrieval-benchmark-revision",
        dest="retrieval_benchmark_revision",
        default=None,
    )
    parser.add_argument(
        "--retrieval-batch-size", dest="retrieval_batch_size", type=int, default=None
    )
    parser.add_argument(
        "--token-rank-tokens", dest="token_rank_tokens", type=int, default=None
    )
    parser.add_argument(
        "--dataloader-workers",
        dest="dataloader_workers",
        type=int,
        default=None,
        help="DataLoader worker processes per training worker",
    )
    parser.add_argument("--log-every", dest="log_every", type=int, default=None)
    parser.add_argument("--eval-every", dest="eval_every", type=int, default=None)
    parser.add_argument("--eval-batches", dest="eval_batches", type=int, default=None)
    parser.add_argument("--save-every", dest="save_every", type=int, default=None)
    parser.add_argument(
        "--full-eval",
        dest="full_eval",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--eval-pool-size", dest="eval_pool_size", type=int, default=None
    )
    parser.add_argument(
        "--eval-counterfactual",
        dest="eval_counterfactual",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="bfloat16 autocast (default: enabled when training on GPU)",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--num-training-workers",
        dest="num_training_workers",
        type=int,
        default=None,
        help="Number of Ray Train workers (1 = single GPU; >1 enables DDP)",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require GPUs for the training workers (default: auto-detect CUDA)",
    )
    parser.add_argument(
        "--storage-path",
        dest="storage_path",
        default=None,
        help="Ray Train results/checkpoint storage path (shared filesystem on "
        "multi-node clusters)",
    )
    parser.add_argument(
        "--checkpoints-to-keep",
        dest="checkpoints_to_keep",
        type=int,
        default=None,
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    trainer = build_trainer(config)
    result = trainer.fit()
    print(f"final metrics: {result.metrics}")
    if result.checkpoint is not None:
        print(f"checkpoint: {result.checkpoint}")
    return


if __name__ == "__main__":
    main()
