from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from bdd import given, then, when
from src.run import train_ray
from src.run.train_ray import (
    RayTrainingConfig,
    build_run_config,
    build_scaling_config,
    build_trainer,
)


def make_config(**overrides) -> RayTrainingConfig:
    return RayTrainingConfig(**overrides)


class TestBuildScalingConfig:
    def test_maps_workers_and_gpu(self):
        with given("a config with two CPU training workers"):
            config = make_config(num_training_workers=2, use_gpu=False)

        with when("the scaling config is built"):
            scaling = build_scaling_config(config)

        with then("it is a ScalingConfig with the same worker and GPU settings"):
            assert isinstance(scaling, ScalingConfig)
            assert scaling.num_workers == 2
            assert scaling.use_gpu is False
        return

    def test_gpu_workers(self):
        with given("a config with one GPU training worker"):
            config = make_config(num_training_workers=1, use_gpu=True)

        with when("the scaling config is built"):
            scaling = build_scaling_config(config)

        with then("use_gpu is enabled"):
            assert scaling.use_gpu is True
        return


class TestBuildRunConfig:
    def test_name_and_storage_path(self, tmp_path):
        with given("a config with a run name and a storage path"):
            config = make_config(
                run_name="exp1", storage_path=str(tmp_path / "results")
            )

        with when("the run config is built"):
            run_config = build_run_config(config)

        with then("the run name and storage path are used"):
            assert isinstance(run_config, RunConfig)
            assert run_config.name == "exp1"
            assert str(run_config.storage_path) == str(tmp_path / "results")
        return

    def test_defaults(self):
        with given("a config without a run name or storage path"):
            config = make_config()

        with when("the run config is built"):
            run_config = build_run_config(config)

        with then("the default run name and storage path are used"):
            assert run_config.name == "chesspos-ray"
            assert str(run_config.storage_path) == "ray_results"
        return

    def test_checkpoints_to_keep(self):
        with given("a config keeping 3 checkpoints"):
            config = make_config(checkpoints_to_keep=3)

        with when("the run config is built"):
            run_config = build_run_config(config)

        with then("the checkpoint config keeps 3 checkpoints"):
            assert run_config.checkpoint_config.num_to_keep == 3
        return


class TestBuildTrainer:
    def test_builds_torch_trainer(self):
        with given("a training config with two GPU workers"):
            config = make_config(num_training_workers=2, use_gpu=True)

        with when("the trainer is built"):
            trainer = build_trainer(config)

        with then("it is a TorchTrainer wired to the config and the training loop"):
            assert isinstance(trainer, TorchTrainer)
            assert trainer.train_loop_per_worker is train_ray.train_loop_per_worker
            assert trainer.train_loop_config == config.to_dict()
            assert trainer.scaling_config.num_workers == 2
            assert trainer.scaling_config.use_gpu is True
        return
