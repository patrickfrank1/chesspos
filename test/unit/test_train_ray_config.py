import pytest
from pathlib import Path

from bdd import given, then, when
from src.modeling.transformer import TransformerConfig
from src.run.train_ray import (
    RayTrainingConfig,
    build_run_config,
    load_yaml_overrides,
    parse_args,
    resolve_config,
)


class TestParseArgs:
    def test_defaults_are_unset(self):
        with given("no CLI arguments"):
            argv: list[str] = []

        with when("the arguments are parsed"):
            args = parse_args(argv)

        with then("the resolution-controlled options are unset (None)"):
            assert args.data_repo is None
            assert args.steps is None
            assert args.batch_size is None
            assert args.use_gpu is None
            assert args.full_eval is None
            assert args.bf16 is None
            assert args.config is None
        return


class TestResolveConfig:
    def test_defaults(self, monkeypatch):
        with given("no CLI arguments, no YAML, and no CUDA device"):
            monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        with when("the config is resolved"):
            config = resolve_config(parse_args([]))

        with then("the defaults match the current training script"):
            assert config.steps == 10_000
            assert config.batch_size == 8
            assert config.lr == 3e-4
            assert config.weight_decay == 0.01
            assert config.warmup == 500
            assert config.grad_clip == 1.0
            assert config.d_model == 512
            assert config.n_layers == 8
            assert config.n_heads == 16
            assert config.d_ff == 2048
            assert config.dropout == 0.1
            assert config.max_seq_len == 2048
            assert config.max_segments == 128
            assert config.dataloader_workers == 4
            assert config.log_every == 20
            assert config.eval_every == 1_000
            assert config.eval_batches == 20
            assert config.save_every == 2_000
            assert config.full_eval is True
            assert config.eval_pool_size == 1024
            assert config.eval_counterfactual == 256
            assert config.bf16 is False
            assert config.compile is False
            assert config.seed == 17
            assert config.num_training_workers == 1
            assert config.use_gpu is False
            assert config.upload_prefix == "checkpoints"
            assert config.checkpoint_dir == "data/models"
            assert config.mlflow_experiment == "chesspos"
            assert config.data_repo is None
        return

    def test_yaml_overrides_defaults(self, tmp_path):
        with given("a YAML config with a training section"):
            yaml_path = tmp_path / "training.yaml"
            yaml_path.write_text(
                "training:\n"
                "  steps: 500\n"
                "  batch_size: 4\n"
                "  data_repo: user/chess-positions\n"
            )

        with when("the config is resolved from the YAML alone"):
            config = resolve_config(parse_args(["--config", str(yaml_path)]))

        with then("the YAML values win over the defaults"):
            assert config.steps == 500
            assert config.batch_size == 4
            assert config.data_repo == "user/chess-positions"
        return

    def test_cli_overrides_yaml(self, tmp_path):
        with given("a YAML config and CLI arguments for the same option"):
            yaml_path = tmp_path / "training.yaml"
            yaml_path.write_text("training:\n  steps: 500\n")

        with when("the config is resolved with --steps on the CLI"):
            config = resolve_config(
                parse_args(["--config", str(yaml_path), "--steps", "777"])
            )

        with then("the CLI value wins"):
            assert config.steps == 777
        return

    def test_use_gpu_auto_detects_cuda(self, monkeypatch):
        with given("no --use-gpu flag and an available CUDA device"):
            monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        with when("the config is resolved"):
            config = resolve_config(parse_args([]))

        with then("use_gpu follows CUDA availability"):
            assert config.use_gpu is True
        return

    def test_bf16_defaults_to_use_gpu(self, monkeypatch):
        with given("no --bf16 flag on a CUDA machine"):
            monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        with when("the config is resolved"):
            config = resolve_config(parse_args([]))

        with then("bf16 is enabled automatically"):
            assert config.bf16 is True
        return

    def test_unknown_yaml_keys_are_ignored(self, tmp_path):
        with given("a YAML config with an unrecognized key"):
            yaml_path = tmp_path / "training.yaml"
            yaml_path.write_text("training:\n  steps: 500\n  bogus: 1\n")

        with when("the config is resolved"):
            config = resolve_config(parse_args(["--config", str(yaml_path)]))

        with then("the unknown key is ignored and the rest applies"):
            assert config.steps == 500
            assert not hasattr(config, "bogus")
        return

    def test_missing_training_section_raises(self, tmp_path):
        with given("a YAML config without a training section"):
            yaml_path = tmp_path / "training.yaml"
            yaml_path.write_text("something_else:\n  steps: 500\n")

        with when("the YAML overrides are loaded"):
            with pytest.raises(ValueError):
                load_yaml_overrides(yaml_path)

        with then("a ValueError is raised"):
            pass
        return


class TestRayTrainingConfig:
    def test_from_dict_ignores_unknown_keys(self):
        with given("a config built from a dict with an unknown key"):
            values = {"steps": 3, "bogus": 1}

        with when("from_dict is called"):
            config = RayTrainingConfig.from_dict(values)

        with then("the known keys apply and the unknown one is dropped"):
            assert config.steps == 3
            assert not hasattr(config, "bogus")
        return

    def test_to_dict_roundtrip(self):
        with given("a config with explicit values"):
            config = RayTrainingConfig(steps=9, data_repo="user/chess-positions")

        with when("it is converted to a dict and back"):
            restored = RayTrainingConfig.from_dict(config.to_dict())

        with then("the configs are equal"):
            assert restored == config
        return

    def test_model_config_property(self):
        with given("a config with model hyperparameters"):
            config = RayTrainingConfig(
                d_model=32,
                n_layers=1,
                n_heads=2,
                d_ff=64,
                dropout=0.2,
                max_seq_len=128,
                max_segments=16,
            )

        with when("the model config is built"):
            model_config = config.model_config

        with then("it is a TransformerConfig with the same hyperparameters"):
            assert model_config == TransformerConfig(
                d_model=32,
                n_layers=1,
                n_heads=2,
                d_ff=64,
                dropout=0.2,
                max_seq_len=128,
                max_segments=16,
            )
        return


class TestBuildRunConfig:
    def test_storage_path_defaults_to_absolute_local_path(self, monkeypatch, tmp_path):
        with given("a config with no storage_path set and a working directory"):
            monkeypatch.chdir(tmp_path)
            config = RayTrainingConfig()

        with when("the run config is built"):
            run_config = build_run_config(config)

        with then("the storage path is an absolute local path under the cwd"):
            storage_path = Path(run_config.storage_path)
            assert storage_path.is_absolute()
            assert storage_path == tmp_path / "ray_results"
        return

    def test_relative_storage_path_is_resolved_against_cwd(self, monkeypatch, tmp_path):
        with given("a config with a relative storage_path and a working directory"):
            monkeypatch.chdir(tmp_path)
            config = RayTrainingConfig(storage_path="runs/storage")

        with when("the run config is built"):
            run_config = build_run_config(config)

        with then("the storage path is resolved against the cwd"):
            storage_path = Path(run_config.storage_path)
            assert storage_path.is_absolute()
            assert storage_path == tmp_path / "runs" / "storage"
        return

    def test_absolute_storage_path_is_kept(self):
        with given("a config with an absolute storage_path"):
            config = RayTrainingConfig(storage_path="/tmp/ray-results")

        with when("the run config is built"):
            run_config = build_run_config(config)

        with then("the storage path is unchanged"):
            assert Path(run_config.storage_path) == Path("/tmp/ray-results")
        return


class TestWorkerRuntimeEnv:
    def test_hf_token_is_passed_to_workers(self, monkeypatch):
        with given("HF_TOKEN is set in the driver environment"):
            monkeypatch.setenv("HF_TOKEN", "secret-token")

        with when("the run config is built"):
            run_config = build_run_config(RayTrainingConfig())

        with then("the worker runtime env carries the token"):
            env_vars = run_config.worker_runtime_env["env_vars"]
            assert env_vars["HF_TOKEN"] == "secret-token"
        return

    def test_unset_hf_vars_yield_no_runtime_env(self, monkeypatch):
        with given("no HF-related variables are set in the driver environment"):
            for name in (
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HUGGING_FACE_HUB_CACHE",
                "HF_HUB_OFFLINE",
            ):
                monkeypatch.delenv(name, raising=False)

        with when("the run config is built"):
            run_config = build_run_config(RayTrainingConfig())

        with then("no worker runtime env override is set"):
            assert not run_config.worker_runtime_env
        return

    def test_unrelated_vars_are_not_passed(self, monkeypatch):
        with given("an unrelated environment variable is set"):
            monkeypatch.setenv("BOGUS_SECRET", "nope")

        with when("the run config is built"):
            run_config = build_run_config(RayTrainingConfig())

        with then("the unrelated variable is not forwarded to workers"):
            env_vars = (run_config.worker_runtime_env or {}).get("env_vars", {})
            assert "BOGUS_SECRET" not in env_vars
        return
