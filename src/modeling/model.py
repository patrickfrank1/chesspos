import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
from keras_nlp import layers as nlp_layers


def get_model(model: str) -> dict[str, keras.Model]:
    if model == "cnn_dense":
        return cnn_dense()
    elif model == "vanilla_dense":
        return vanilla_dense()
    elif model == "trivial":
        return trivial()
    elif model == "skip_dense":
        return skip_dense()
    elif model == "skip_equi_dense":
        return skip_equi_dense()
    elif model == "encoder_decoder_transformer":
        return encoder_decoder_transformer()
    else:
        raise ValueError("The requested neral network architecture does not exist.")


def vanilla_dense() -> dict[str, keras.Model]:
    EMBEDDING_SIZE = 256
    dtype = tf.bfloat16

    # Encoder
    encoder_input = layers.Input(shape=(8, 8, 15), dtype=dtype)
    encoder = layers.Reshape((8*8*15,))(encoder_input)
    encoder = layers.Dense(8*EMBEDDING_SIZE, activation='relu')(encoder)
    encoder = layers.Dense(4*EMBEDDING_SIZE, activation='relu')(encoder)
    encoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(encoder)
    encoder = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)

    # Decoder
    decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
    decoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(decoder_input)
    decoder = layers.Dense(4*EMBEDDING_SIZE, activation='relu')(decoder_input)
    decoder = layers.Dense(8*EMBEDDING_SIZE, activation='relu')(decoder_input)
    decoder = layers.Dense(8*8*15, activation='relu')(decoder)
    decoder = layers.Reshape((8, 8, 15))(decoder)

    # Autoencoder
    encoder = keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')
    decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

    return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}


def skip_dense() -> dict[str, keras.Model]:
    EMBEDDING_SIZE = 768
    dtype = tf.bfloat16

    # Encoder
    encoder_input = layers.Input(shape=(8, 8, 15), dtype=dtype)
    encoder = layers.Reshape((8*8*15,))(encoder_input)
    encoder = layers.Dense(8*EMBEDDING_SIZE, activation='relu')(encoder)
    encoder_skip_1 = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)
    encoder = layers.Dense(4*EMBEDDING_SIZE, activation='relu')(encoder)
    encoder_skip_2 = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)
    encoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(encoder)
    encoder_skip_3 = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)
    embedding = layers.add([encoder_skip_1, encoder_skip_2, encoder_skip_3])

    # Decoder
    decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
    decoder_skip_1 = layers.Dense(8*EMBEDDING_SIZE, activation='relu')(decoder_input)
    decoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(decoder_input)
    decoder_skip_2 = layers.Dense(8*EMBEDDING_SIZE, activation='relu')(decoder)
    decoder = layers.Dense(4*EMBEDDING_SIZE, activation='relu')(decoder)
    decoder_skip_3 = layers.Dense(8*EMBEDDING_SIZE, activation='relu')(decoder)
    decoder = layers.add([decoder_skip_1, decoder_skip_2, decoder_skip_3])
    decoder = layers.Dense(8*8*15, activation='relu')(decoder)
    decoder = layers.Reshape((8, 8, 15))(decoder)

    # Autoencoder
    encoder = keras.Model(inputs=encoder_input, outputs=embedding, name='encoder')
    decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

    return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}


def skip_equi_dense() -> dict[str, keras.Model]:
    EMBEDDING_SIZE = 2560
    BLOCKS = 2
    dtype = tf.bfloat16

    def block_with_skip_connection(previous_layer):
        embedding_layer = layers.Dense(EMBEDDING_SIZE, activation='relu')(previous_layer)
        combine_layer = layers.add([previous_layer, embedding_layer])
        return combine_layer

    # Encoder
    encoder_input = layers.Input(shape=(8, 8, 15), dtype=dtype)
    encoder = layers.Reshape((8*8*15,))(encoder_input)
    encoder = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)
    for _ in range(BLOCKS):
        encoder = block_with_skip_connection(encoder)

    # Decoder
    decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
    decoder = decoder_input
    for _ in range(BLOCKS):
        decoder = block_with_skip_connection(decoder)
    decoder = layers.Dense(8*8*15, activation='relu')(decoder)
    decoder = layers.Reshape((8, 8, 15))(decoder)

    # Autoencoder
    encoder = keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')
    decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

    return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}


def cnn_dense() -> dict[str, keras.Model]:
    EMBEDDING_SIZE = 64
    CONV_FILTERS = 32
    dtype = tf.bfloat16

    encoder_input = layers.Input(shape=(8, 8, 15, 1), dtype=dtype)

    # Encoder
    x = layers.Conv3D(CONV_FILTERS, (8, 8, 15), activation="relu", padding="same")(encoder_input)
    x = layers.Conv3D(CONV_FILTERS, (8, 8, 15), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
    x = layers.Conv3D(2*CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(x)
    x = layers.Conv3D(2*CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
    x = layers.Conv3D(4*CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(x)
    x = layers.Conv3D(4*CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(EMBEDDING_SIZE, activation="relu")(x)

    # Decoder
    decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
    y = layers.Dense(4*CONV_FILTERS*2*2*15, activation="relu")(decoder_input)
    y = layers.Reshape((2, 2, 15, 4*CONV_FILTERS))(y)
    y = layers.Conv3D(4*CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(y)
    y = layers.Conv3DTranspose(2*CONV_FILTERS, (3, 3, 15), strides=(2, 2, 1), activation="relu", padding="same")(y)
    y = layers.Conv3D(2*CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(y)
    y = layers.Conv3DTranspose(CONV_FILTERS, (3, 3, 15), strides=(2, 2, 1), activation="relu", padding="same")(y)
    y = layers.Conv3D(CONV_FILTERS, (8, 8, 15), activation="relu", padding="same")(y)
    y = layers.Conv3D(1, (8, 8, 15), activation="relu", padding="same")(y)
    y = layers.Reshape((8, 8, 15))(y)

    # Autoencoder
    encoder = keras.Model(inputs=encoder_input, outputs=x, name='encoder')
    decoder = keras.Model(inputs=decoder_input, outputs=y, name='decoder')
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

    return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}


def trivial() -> dict[str, keras.Model]:
    EMBEDDING_SIZE = 5120
    dtype = tf.bfloat16

    # Encoder
    encoder_input = layers.Input(shape=(8, 8, 15), dtype=dtype)
    encoder = layers.Reshape((8*8*15,))(encoder_input)
    encoder_embedding = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)

    # Decoder
    decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
    decoder = layers.Dense(8*8*15, activation='relu')(decoder_input)
    decoder = layers.Reshape((8, 8, 15))(decoder)

    # Autoencoder
    encoder = keras.Model(inputs=encoder_input, outputs=encoder_embedding, name='encoder')
    decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

    return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}


def encoder_decoder_transformer() -> dict[str, keras.Model]:
    ENCODING_DTYPE = "int8"
    NN_DTYPE = "bfloat16"
    SEQUENCE_LENGTH = 69
    VOCABULARY_SIZE = 33
    INTERMEDIATE_DIMENSION = 512
    EMBEDDING_DIMENSION = 256
    DROPOUT = 0.1
    NUM_HEADS = 16
    ENCODER_LAYERS = 2
    DECODER_LAYERS = 2
    STD_DEV = 0.02

    # Encoder
    encoder_token_ids = layers.Input(shape=(SEQUENCE_LENGTH), dtype=ENCODING_DTYPE, name="encoder_token_ids")

    # Embed tokens ans positions
    token_embedding_layer = nlp_layers.ReversibleEmbedding(
        input_dim=VOCABULARY_SIZE,
        output_dim=EMBEDDING_DIMENSION,
        embeddings_initializer=keras.initializers.TruncatedNormal(stddev=STD_DEV),
        name="token_embedding",
    )
    token_embedding = token_embedding_layer(encoder_token_ids)

    position_embedding = nlp_layers.PositionEmbedding(
        initializer=keras.initializers.TruncatedNormal(stddev=STD_DEV),
        sequence_length=SEQUENCE_LENGTH,
        name="position_embedding",
    )(token_embedding)

    # Sum, normalize and apply dropout to embeddings.
    x = layers.Add()([token_embedding, position_embedding])
    x = layers.LayerNormalization(
        name="embeddings_layer_norm",
        axis=-1,
        epsilon=1e-12,
        dtype=NN_DTYPE,
    )(x)
    x = layers.Dropout(DROPOUT, name="embeddings_dropout",)(x)

    # Apply successive transformer encoder blocks.
    for i in range(ENCODER_LAYERS):
        x = nlp_layers.TransformerEncoder(
            num_heads=NUM_HEADS,
            intermediate_dim=INTERMEDIATE_DIMENSION,
            activation=lambda x: keras.activations.gelu(x, approximate=True),
            dropout=DROPOUT,
            layer_norm_epsilon=1e-12,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=STD_DEV),
            name=f"transformer_layer_{i}",
        )(x)

    x = layers.Flatten()(x)
    encoder_embedding = layers.Dense(
        EMBEDDING_DIMENSION,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=STD_DEV),
        activation="tanh",
        name="dense"
    )(x)

    # Decoder
    decoder_input = layers.Input(shape=(EMBEDDING_DIMENSION), dtype=NN_DTYPE, name="decoder_input")
    x = layers.Dense(
        SEQUENCE_LENGTH*EMBEDDING_DIMENSION,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=STD_DEV),
        activation="tanh",
        name="dense"
    )(decoder_input)
    x = layers.Reshape((SEQUENCE_LENGTH, EMBEDDING_DIMENSION))(x)
    # x = keras.layers.Dropout(
    # 	DROPOUT,
    # 	name="embeddings_dropout",
    # )(decoder_input)

    # Apply successive transformer decoder blocks.
    for i in range(DECODER_LAYERS):
        x = nlp_layers.TransformerDecoder(
            intermediate_dim=INTERMEDIATE_DIMENSION,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            layer_norm_epsilon=1e-05,
            activation=lambda x: keras.activations.gelu(x, approximate=True),
            kernel_initializer=keras.initializers.RandomNormal(stddev=STD_DEV),
            normalize_first=True,
            name=f"transformer_layer_{i}",
        )(x)

    decoder = keras.layers.LayerNormalization(
        name="layer_norm",
        axis=-1,
        epsilon=1e-05,
        dtype=NN_DTYPE,
    )(x)
    decoder_logits = token_embedding_layer(decoder, reverse=True)

    # Autoencoder
    encoder = keras.Model(inputs=encoder_token_ids, outputs=encoder_embedding, name='encoder')
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_logits, name='decoder')
    autoencoder = keras.Model(inputs=encoder_token_ids, outputs=decoder(encoder(encoder_token_ids)), name='autoencoder')

    return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}


class PositionPredictionHead():
    def __init__(self, backend: keras.Model):
        self.backend = backend

    def predict_on_batch(self, inputs):
        logits = self.backend.predict_on_batch(inputs)
        probabilities = K.softmax(logits, axis=2)
        predicted_tokens = K.argmax(probabilities, axis=2)
        return predicted_tokens

    def evaluate(self, x):
        y = self.predict_on_batch(x)
        diff = tf.subtract(y, x)
        mispredicted = tf.math.count_nonzero(diff)
        return mispredicted
