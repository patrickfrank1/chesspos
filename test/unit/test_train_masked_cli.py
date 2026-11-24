import numpy as np
import pytest

from bdd import given, then, when
from src.dataset.token_stream import CLS, EMPTY, PIECE_BASE, VOCAB_SIZE
from src.modeling.autoencoder import AttentionAutoencoderConfig
from src.modeling.factory import build_model
from src.modeling.transformer import TransformerConfig
from src.modeling.masking import MaskingConfig
from src.training.window_dataset import balanced_v1_weights
from src.run.train_masked_transformer import (
    auxiliary_config,
    dataset_token_weights,
    describe_token_weights,
    full_reconstruction_weights,
    masking_config,
    parse_args,
    streaming_train_config,
    view_helpers,
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

    def test_content_pooling_is_enabled_by_default(self, monkeypatch):
        with given("no content-pooling flag"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])

        with when("the arguments are parsed"):
            args = parse_args()

        with then("the strategic content pool is enabled"):
            assert args.content_pooling is True
        return

    def test_content_pooling_can_be_disabled(self, monkeypatch):
        with given("the --no-content-pooling flag"):
            monkeypatch.setattr(
                "sys.argv", ["train_masked_transformer", "--no-content-pooling"]
            )

        with when("the arguments are parsed"):
            args = parse_args()

        with then("mean pooling is selected"):
            assert args.content_pooling is False
        return

    def test_validation_pool_is_bounded_by_default(self, monkeypatch):
        with given("no CLI arguments"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])

        with when("the arguments are parsed"):
            args = parse_args()

        with then("the validation pool is capped rather than loading the test split"):
            assert args.eval_pool_games == 8192
        return


class TestLossWeights:
    def test_visible_identity_weight_is_carried_to_loss_config(self, monkeypatch):
        with given("a visible-identity CLI weight"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--visible-identity-weight", "0.25"],
            )
            args = parse_args()

        with when("the auxiliary loss config is built"):
            config = auxiliary_config(args)

        with then("the weight reaches the objective"):
            assert config.visible_identity_weight == 0.25
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

    def test_streaming_train_config_can_disable_resume_skip(self, monkeypatch):
        with given("streaming args with resume data skipping disabled"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--streaming",
                    "--no-resume-data-skip",
                ],
            )
            args = parse_args()

        with when("the streaming train config is built for a resumed run"):
            config = streaming_train_config(
                args, data_dir="data/train", start_step=5000
            )

        with then("the data stream starts from the beginning"):
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
            assert config.p_rect == 0.34
            assert config.p_random == 0.0
            assert config.p_board == 0.0
            assert config.p_span == 0.0
            assert config.p_skip_gram == 0.66
            assert config.skip_gram_unmask_rate == reference.skip_gram_unmask_rate
            assert config.rect_min_rate == reference.rect_min_rate
            assert config.rect_max_rate == reference.rect_max_rate
            assert config.rect_min_side == reference.rect_min_side
            assert config.rect_max_side == reference.rect_max_side
            assert config.rect_synchronize is True
        return

    def test_skip_gram_flag_disables_rect(self, monkeypatch):
        with given("skip-gram masking enabled without an explicit p_rect"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--p-skip-gram", "1.0"],
            )
            args = parse_args()

        with when("the masking config is built"):
            config = masking_config(args)

        with then("skip-gram is selected and rectangle masking is off"):
            assert config.p_skip_gram == 1.0
            assert config.p_rect == 0.0
            assert config.skip_gram_unmask_rate == 0.10
        return

    def test_skip_gram_rate_override(self, monkeypatch):
        with given("an explicit skip-gram visible rate"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--p-skip-gram",
                    "1.0",
                    "--skip-gram-unmask-rate",
                    "0.2",
                ],
            )
            args = parse_args()

        with when("the masking config is built"):
            config = masking_config(args)

        with then("the rate override is carried"):
            assert config.skip_gram_unmask_rate == 0.2
        return

    def test_explicit_rect_skip_gram_mixture_is_preserved(self, monkeypatch):
        with given("explicit mixture weights"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--p-rect",
                    "0.8",
                    "--p-skip-gram",
                    "0.2",
                ],
            )
            args = parse_args()

        with when("the masking config is built"):
            config = masking_config(args)

        with then("both weights are kept"):
            assert config.p_rect == 0.8
            assert config.p_skip_gram == 0.2
        return

    def test_explicit_rect_disables_skip_gram(self, monkeypatch):
        with given("only an explicit rectangle weight"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--p-rect", "1.0"],
            )
            args = parse_args()

        with when("the masking config is built"):
            config = masking_config(args)

        with then("skip-gram is off"):
            assert config.p_rect == 1.0
            assert config.p_skip_gram == 0.0
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
    def test_default_weights_enable_geometry_and_nce(self, monkeypatch):
        with given("no loss weight flags"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])
            args = parse_args()

        with when("the auxiliary config is built"):
            config = auxiliary_config(args)

        with then("strong geometry, invariance and InfoNCE are on"):
            assert config.mlm_weight == 1.0
            assert config.inv_weight == 0.5
            assert config.var_weight == 1.0
            assert config.cov_weight == 0.1
            assert config.nce_weight == 0.1
            assert config.alone_var_weight == 1.0
            assert config.alone_cov_weight == 0.1
            assert config.alone_targets_per_window == 4
            assert config.warmup_steps == 200
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
                    "--alone-var-weight",
                    "0",
                    "--alone-cov-weight",
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

    def test_alone_vicreg_flags_override(self, monkeypatch):
        with given("explicit alone-view weights and target count"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--alone-var-weight",
                    "0.4",
                    "--alone-cov-weight",
                    "0.04",
                    "--alone-targets-per-window",
                    "2",
                ],
            )
            args = parse_args()

        with when("the auxiliary config is built"):
            config = auxiliary_config(args)

        with then("the alone-view settings reach the config"):
            assert config.alone_var_weight == 0.4
            assert config.alone_cov_weight == 0.04
            assert config.alone_targets_per_window == 2
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
            assert config.p_skip_gram == 0.0
            assert config.random_replace_prob == MaskingConfig().random_replace_prob
        return


class TestStreamShardsFlags:
    def test_stream_shards_defaults_off(self, monkeypatch):
        with given("no CLI arguments"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])

        with when("the arguments are parsed"):
            args = parse_args()

        with then("lazy shard fetching is disabled with a default cache dir"):
            assert args.stream_shards is False
            assert args.stream_shard_cache == "data/hf-shards"
        return

    def test_stream_shards_flags_are_parsed(self, monkeypatch):
        with given("the lazy shard flags"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--stream-shards",
                    "--stream-shard-cache",
                    "/tmp/shards",
                ],
            )

        with when("the arguments are parsed"):
            args = parse_args()

        with then("the flag and cache directory are carried through"):
            assert args.stream_shards is True
            assert args.stream_shard_cache == "/tmp/shards"
        return

    def test_streaming_config_carries_remote_shard_settings(self, monkeypatch):
        with given("streaming args with lazy shard fetching"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--streaming",
                    "--stream-shards",
                    "--data-repo",
                    "user/chesspos",
                    "--stream-shard-cache",
                    "/tmp/shards",
                ],
            )
            args = parse_args()

        with when("the streaming train config is built"):
            config = streaming_train_config(args, data_dir="/tmp/shards", start_step=0)

        with then("the config targets the hub repo with lazy delete enabled"):
            assert config.remote_repo == "user/chesspos"
            assert config.remote_split == "train"
            assert config.remote_cache_dir == "/tmp/shards"
            assert config.remote_delete is True
        return

    def test_streaming_config_has_no_remote_shards_by_default(self, monkeypatch):
        with given("streaming args without lazy shard fetching"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--streaming", "--data-repo", "u/d"],
            )
            args = parse_args()

        with when("the streaming train config is built"):
            config = streaming_train_config(args, data_dir="data/train", start_step=0)

        with then("remote shard fetching stays disabled"):
            assert config.remote_repo is None
        return


class TestAttentionAutoencoderFlags:
    def test_autoencoder_preset_matches_the_deep_narrow_recipe(self, monkeypatch):
        with given("the attention autoencoder architecture and no size flags"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--model-arch", "attention_autoencoder"],
            )

        with when("the arguments are parsed"):
            args = parse_args()

        with then("the deep, narrow-bottleneck preset is selected"):
            assert args.d_model == 512
            assert args.n_heads == 8
            assert args.n_layers == 6
            assert args.d_ff == 1024
            assert args.n_decoder_layers == 3
            assert args.latent_dim == 512
        return

    def test_explicit_size_overrides_the_preset(self, monkeypatch):
        with given("the autoencoder architecture with explicit widths"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--model-arch",
                    "attention_autoencoder",
                    "--d-model",
                    "320",
                    "--n-layers",
                    "5",
                    "--n-heads",
                    "8",
                    "--d-ff",
                    "1280",
                    "--n-decoder-layers",
                    "2",
                    "--latent-dim",
                    "768",
                ],
            )

        with when("the arguments are parsed"):
            args = parse_args()

        with then("the explicit values win over the preset"):
            assert args.d_model == 320
            assert args.n_layers == 5
            assert args.n_heads == 8
            assert args.d_ff == 1280
            assert args.n_decoder_layers == 2
            assert args.latent_dim == 768
        return

    def test_autoencoder_auxiliary_config_is_reconstruction_only(self, monkeypatch):
        with given("the autoencoder architecture and default loss weights"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--model-arch", "attention_autoencoder"],
            )
            args = parse_args()

        with when("the auxiliary config is built"):
            config = auxiliary_config(args)

        with then("only the reconstruction loss is active"):
            assert config.auxiliary_enabled() is False
            assert config.mlm_weight == 1.0
            assert config.warmup_steps == 0
        return

    def test_autoencoder_weights_every_position(self, monkeypatch):
        with given("the autoencoder architecture"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--model-arch", "attention_autoencoder"],
            )
            args = parse_args()

        with when("the weighting policy is queried"):
            full = full_reconstruction_weights(args)

        with then("visible positions stay supervised"):
            assert full is True
        return

    def test_transformer_keeps_masked_only_weighting(self, monkeypatch):
        with given("the default transformer architecture"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])
            args = parse_args()

        with when("the weighting policy is queried"):
            full = full_reconstruction_weights(args)

        with then("only masked positions are supervised"):
            assert full is False
        return

    def test_streaming_config_carries_reconstruction_weighting(self, monkeypatch):
        with given("streaming autoencoder args"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--streaming",
                    "--model-arch",
                    "attention_autoencoder",
                ],
            )
            args = parse_args()

        with when("the streaming train config is built"):
            config = streaming_train_config(args, data_dir="data/train", start_step=0)

        with then("full-sequence weighting reaches the dataset"):
            assert config.full_reconstruction_weights is True
        return


class TestViewHelpers:
    def test_transformer_without_identity_loss_pools_views(self, monkeypatch):
        with given("a transformer model and the default objectives"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])
            args = parse_args()
            model = build_model(TransformerConfig(d_model=16, n_heads=2, n_layers=1))

        with when("the view helpers are resolved"):
            content_hidden, encode_views, alone = view_helpers(model, args)

        with then("the content-only helper stays unset and views are pooled"):
            assert content_hidden is None
            assert encode_views is not None
            assert alone is not None
        return

    def test_identity_objective_uses_the_content_trunk(self, monkeypatch):
        with given("a transformer model and a positive visible-identity weight"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--visible-identity-weight", "0.5"],
            )
            args = parse_args()
            model = build_model(TransformerConfig(d_model=16, n_heads=2, n_layers=1))

        with when("the view helpers are resolved"):
            content_hidden, encode_views, alone = view_helpers(model, args)

        with then("the content helper is used and the view pools are disabled"):
            assert content_hidden is not None
            assert encode_views is None
            assert alone is not None
        return

    def test_autoencoder_has_no_view_helpers(self, monkeypatch):
        with given("an autoencoder model without encode_views"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--model-arch", "attention_autoencoder"],
            )
            args = parse_args()
            model = build_model(
                AttentionAutoencoderConfig(
                    d_model=16,
                    n_heads=2,
                    n_layers=1,
                    d_ff=32,
                    latent_dim=8,
                    n_decoder_layers=1,
                )
            )

        with when("the view helpers are resolved"):
            content_hidden, encode_views, alone = view_helpers(model, args)

        with then("every view helper is unset instead of raising"):
            assert content_hidden is None
            assert encode_views is None
            assert alone is None
        return


class TestTokenWeightPreset:
    def test_default_preset_is_inverse_frequency(self, monkeypatch):
        with given("no preset flag"):
            monkeypatch.setattr("sys.argv", ["train_masked_transformer"])

        with when("the arguments are parsed"):
            args = parse_args()

        with then("the historical inverse-frequency scheme is used"):
            assert args.token_weight_preset == "invfreq"
        return

    def test_balanced_v1_skips_unigram_counting(self, monkeypatch):
        with given("the balanced-v1 preset"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--token-weight-preset", "balanced-v1"],
            )
            args = parse_args()
            calls = []

            def counts_supplier():
                calls.append(1)
                return np.ones(VOCAB_SIZE)

        with when("the dataset token weights are resolved"):
            weights = dataset_token_weights(args, counts_supplier=counts_supplier)

        with then("the fixed table is used without touching the counts"):
            assert calls == []
            assert weights is not None
            assert weights[EMPTY] == pytest.approx(1.0)
            assert weights[CLS] == 0.0
        return

    def test_invfreq_preset_uses_counts_and_cap(self, monkeypatch):
        with given("the default preset with --freq-cap 5"):
            monkeypatch.setattr(
                "sys.argv", ["train_masked_transformer", "--freq-cap", "5"]
            )
            args = parse_args()
            counts = np.zeros(VOCAB_SIZE)
            counts[EMPTY] = 1_000_000
            counts[PIECE_BASE] = 10

        with when("the dataset token weights are resolved"):
            weights = dataset_token_weights(args, counts_supplier=lambda: counts)

        with then("the cap is applied to the inverse-frequency table"):
            assert weights is not None
            assert weights[EMPTY] == pytest.approx(1.0)
            assert weights[PIECE_BASE] == pytest.approx(5.0)
        return

    def test_no_freq_weighting_takes_precedence(self, monkeypatch):
        with given("both --no-freq-weighting and a preset"):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "train_masked_transformer",
                    "--no-freq-weighting",
                    "--token-weight-preset",
                    "balanced-v1",
                ],
            )
            args = parse_args()
            calls = []

            def counts_supplier():
                calls.append(1)
                return np.ones(VOCAB_SIZE)

        with when("the dataset token weights are resolved"):
            weights = dataset_token_weights(args, counts_supplier=counts_supplier)

        with then("weighting is disabled without touching the counts"):
            assert weights is None
            assert calls == []
        return

    def test_unknown_preset_is_rejected_by_the_cli(self, monkeypatch):
        with given("an unrecognised preset on the command line"):
            monkeypatch.setattr(
                "sys.argv",
                ["train_masked_transformer", "--token-weight-preset", "mystery"],
            )

        with when("the arguments are parsed"):
            with then("argparse rejects the choice"):
                with pytest.raises(SystemExit):
                    parse_args()
        return

    def test_balanced_v1_summary_reports_the_table(self):
        with given("the resolved balanced-v1 table"):
            weights = balanced_v1_weights()

        with when("the log summary is formatted"):
            text = describe_token_weights("balanced-v1", weights)

        with then("the preset and the key weights are visible"):
            assert "preset=balanced-v1" in text
            assert "empty=1.0" in text
            assert "pawn=2.0" in text
            assert "queen=4.0" in text
            assert "cls=0.0" in text
        return
