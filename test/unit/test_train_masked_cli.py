from bdd import given, then, when
from src.modeling.masking import MaskingConfig
from src.run.train_masked_transformer import (
    auxiliary_config,
    masking_config,
    parse_args,
    streaming_train_config,
)


class TestParseArgs:
    def test_default_architecture_is_scaled_up(self, monkeypatch):
        with given("no CLI arguments"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])

        with when("the arguments are parsed"):
            args = parse_args()

        with then("the default architecture is the scaled-up configuration"):
            assert args.d_model == 512
            assert args.n_heads == 16
            assert args.n_layers == 8
            assert args.d_ff == 2048
        return


class TestStreamingFlag:
    def test_streaming_defaults_to_off(self, monkeypatch):
        with given("no CLI arguments"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])

        with when("the arguments are parsed"):
            args = parse_args()

        with then("streaming mode is disabled"):
            assert args.streaming is False
        return

    def test_streaming_flag_enables_streaming(self, monkeypatch):
        with given("the --streaming flag"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer", "--streaming"])

        with when("the arguments are parsed"):
            args = parse_args()

        with then("streaming mode is enabled"):
            assert args.streaming is True
        return

    def test_streaming_train_config_carries_resume_state(self, monkeypatch):
        with given("streaming args and a checkpoint at step 5000"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--streaming",
                    "--batch-size",
                    "128",
                    "--num-workers",
                    "16",
                    "--seed",
                    "17",
                    "--shuffle-buffer",
                    "4096",
                    "--prefetch-shards",
                    "2",
                ],
            )
            args = parse_args()

        with when("the streaming train config is built for the resumed run"):
            config = streaming_train_config(
                args, data_dir="data/train", start_step=5000
            )

        with then("the config fast-forwards the stream to the checkpoint position"):
            assert config.resume_step == 5000
            assert config.batch_size == 128
            assert config.num_workers == 16
            assert config.seed == 17
            assert config.shuffle_buffer == 4096
            assert config.prefetch_shards == 2
            assert config.data_dir == "data/train"
        return

    def test_streaming_train_config_fresh_start(self, monkeypatch):
        with given("streaming args and no checkpoint"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer", "--streaming"])
            args = parse_args()

        with when("the streaming train config is built for step zero"):
            config = streaming_train_config(args, data_dir="data/train", start_step=0)

        with then("the config starts the stream from the beginning"):
            assert config.resume_step == 0
        return

    def test_masking_probability_flags_exist_with_defaults(self, monkeypatch):
        with given("no masking flags"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])
            args = parse_args()

        with when("the masking config is built from the args"):
            config = masking_config(args)

        with then("rectangle masking is the default and legacy modes are off"):
            reference = MaskingConfig()
            assert config.p_rect == 1.0
            assert config.p_random == 0.0
            assert config.p_board == 0.0
            assert config.p_span == 0.0
            assert config.rect_min_rate == reference.rect_min_rate
            assert config.rect_max_rate == reference.rect_max_rate
            assert config.rect_min_side == reference.rect_min_side
            assert config.rect_max_side == reference.rect_max_side
            assert config.rect_synchronize is True
        return

    def test_rectangle_mask_flags_override(self, monkeypatch):
        with given("explicit rectangle mask flags"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--p-rect",
                    "0.0",
                    "--p-random",
                    "0.5",
                    "--rect-min-rate",
                    "0.35",
                    "--rect-max-rate",
                    "0.45",
                    "--rect-min-side",
                    "3",
                    "--rect-max-side",
                    "5",
                    "--no-rect-sync",
                ],
            )
            args = parse_args()

        with when("the masking config is built from the args"):
            config = masking_config(args)

        with then("the overrides are carried into the config"):
            assert config.p_rect == 0.0
            assert config.p_random == 0.5
            assert config.rect_min_rate == 0.35
            assert config.rect_max_rate == 0.45
            assert config.rect_min_side == 3
            assert config.rect_max_side == 5
            assert config.rect_synchronize is False
        return


class TestAuxiliaryConfig:
    def test_default_weights_enable_geometry_but_not_nce(self, monkeypatch):
        with given("no loss weight flags"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])
            args = parse_args()

        with when("the auxiliary config is built"):
            config = auxiliary_config(args)

        with then("geometry is on and InfoNCE is off"):
            assert config.mlm_weight == 1.0
            assert config.inv_weight == 0.1
            assert config.var_weight == 0.1
            assert config.cov_weight == 0.01
            assert config.nce_weight == 0.0
            assert config.auxiliary_enabled() is True
        return

    def test_all_auxiliary_weights_zero_disables_geometry(self, monkeypatch):
        with given("all geometry and InfoNCE weights set to zero"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--inv-weight",
                    "0",
                    "--var-weight",
                    "0",
                    "--cov-weight",
                    "0",
                    "--nce-weight",
                    "0",
                ],
            )
            args = parse_args()

        with when("the auxiliary config is built"):
            config = auxiliary_config(args)

        with then("the training step is MLM-only"):
            assert config.auxiliary_enabled() is False
        return

    def test_weight_and_warmup_flags_override(self, monkeypatch):
        with given("explicit loss weights and warmup"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--mlm-weight",
                    "2.0",
                    "--inv-weight",
                    "0.2",
                    "--nce-weight",
                    "0.3",
                    "--nce-temperature",
                    "0.07",
                    "--aux-warmup",
                    "100",
                ],
            )
            args = parse_args()

        with when("the auxiliary config is built"):
            config = auxiliary_config(args)

        with then("the weights and warmup are carried into the config"):
            assert config.mlm_weight == 2.0
            assert config.inv_weight == 0.2
            assert config.nce_weight == 0.3
            assert config.nce_temperature == 0.07
            assert config.warmup_steps == 100
        return


class TestRetrievalFlags:
    def test_retrieval_benchmark_flags_are_parsed(self, monkeypatch):
        with given("retrieval benchmark arguments"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--retrieval-benchmark",
                    "patrickfrank1/chess-positions",
                    "--retrieval-benchmark-revision",
                    "main",
                    "--retrieval-batch-size",
                    "64",
                    "--token-rank-tokens",
                    "2048",
                ],
            )
            args = parse_args()

        with when("the arguments are parsed"):
            pass

        with then("the benchmark settings are available"):
            assert args.retrieval_benchmark == "patrickfrank1/chess-positions"
            assert args.retrieval_benchmark_revision == "main"
            assert args.retrieval_batch_size == 64
            assert args.token_rank_tokens == 2048
        return

    def test_masking_probability_flags_override(self, monkeypatch):
        with given("explicit masking probability flags"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--p-random",
                    "0.25",
                    "--p-board",
                    "0.30",
                    "--p-span",
                    "0.45",
                ],
            )
            args = parse_args()

        with when("the masking config is built from the args"):
            config = masking_config(args)

        with then("the config carries the overridden probabilities"):
            assert config.p_random == 0.25
            assert config.p_board == 0.30
            assert config.p_span == 0.45
            assert config.p_rect == 0.0
            assert config.random_replace_prob == MaskingConfig().random_replace_prob
        return
