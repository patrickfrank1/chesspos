from bdd import given, then, when
from src.modeling.masking import MaskingConfig
from src.run.train_masked_transformer import (
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

        with then("the config matches the current MaskingConfig defaults"):
            reference = MaskingConfig()
            assert config.p_random == reference.p_random
            assert config.p_board == reference.p_board
            assert config.p_span == reference.p_span
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
            assert config.random_replace_prob == MaskingConfig().random_replace_prob
        return
