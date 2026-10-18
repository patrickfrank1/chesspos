from bdd import given, then, when
from src.run.train_masked_transformer import parse_args


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
