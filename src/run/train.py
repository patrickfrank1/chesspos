import os
import datetime as dt

import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
import mlflow

from src.types import IOTensorPair
from src.modeling.model import get_model
from src.training.sample_generator import AutoencoderDataGenerator, ReconstructAutoencoderDataGenerator
from src.modeling import custom_losses as cl


if __name__ == "__main__":

    mlflow.autolog()

    # parameters
    DATA_DIR = "./data"
    MODEL_DIR = "./model"
    BATCH_SIZE = 16
    EPOCHS = 20
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100
    MASKED_SQUARES = 10
    MODEL_TYPE = "encoder_decoder_transformer"

    # create required directories if they do not yet exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/checkpoints", exist_ok=True)

    # get model definition
    autoencoder: keras.Model = get_model(MODEL_TYPE)["autoencoder"]

    # compile model
    optimizer = 'rmsprop'
    if MODEL_TYPE == "encoder_decoder_transformer":
        autoencoder.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            jit_compile=True,
        )
    else:
        autoencoder.compile(
            optimizer=optimizer,
            loss=cl.custom_regularized_loss,
            metrics=[
                cl.sum_squared_loss,
                cl.num_pc_reg,
                cl.pc_column_reg,
                cl.pc_plane_reg
            ],
            jit_compile=True
        )

    # plot model graph
    plot_model(autoencoder, to_file=f"{MODEL_DIR}/model_plot.png", show_shapes=True, expand_nested=True)

    # print model architecture
    autoencoder.summary(expand_nested=True)

    # load train and test data
    train_data = ReconstructAutoencoderDataGenerator(
        f"{DATA_DIR}/train",
        number_squares=MASKED_SQUARES,
        batch_size=BATCH_SIZE
    )
    test_data = ReconstructAutoencoderDataGenerator(
        f"{DATA_DIR}/test",
        number_squares=MASKED_SQUARES,
        batch_size=BATCH_SIZE
    )

    # TODO: also print a couple of layers

    def tensor_preview(data: IOTensorPair):
        print(f"Number of samples: {data.total_dataset_length()}")
        train_batch = data.__getitem__(0)
        print(f"First batch: len={len(train_batch)}")
        train_sample = train_batch[0]
        print(f"First item: shape={train_sample.shape}, dtype={train_sample.dtype}")

        pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        piece_map = ["white " + piece for piece in pieces] + \
                    ["black " + piece for piece in pieces] + \
                    ["castling rights", "en passant", "turn"]
        print("First train position:")
        for i, piece in enumerate(piece_map):
            print(piece)
            print(train_sample[0, :, :, i].astype(float))

    def token_preview(data: IOTensorPair):
        print(f"Number of samples: {data.total_dataset_length()}")
        train_batch = data.__getitem__(0)
        print(f"First batch: len={len(train_batch)}")
        train_sample = train_batch[0]
        print(f"First item: shape={train_sample.shape}, dtype={train_sample.dtype}")
        print("First train position:")
        print(train_sample[0].astype(int))

    def test_custom_loss_functions(data: IOTensorPair) -> None:
        sample_1, sample_2 = data.__getitem__(0)
        sample_1 = tf.convert_to_tensor(sample_1[0:1], dtype=tf.int8)
        sample_2 = tf.convert_to_tensor(sample_2[0:1], dtype=tf.int8)

        print("Sum squared loss on applied to first train-test samples:")
        print(cl.sum_squared_loss(sample_1, sample_2))
        print("Total number of pieces loss:")
        print(cl.num_pc_reg(sample_1, sample_2))
        print("Number of pieces per square loss:")
        print(cl.pc_column_reg(sample_1, sample_2))
        print("Number of pieces per plane loss:")
        print(cl.pc_plane_reg(sample_1, sample_2))

    if MODEL_TYPE == "encoder_decoder_transformer":
        preview_function = token_preview
        custom_loss_function = lambda _: "no op"
    else:
        preview_function = tensor_preview
        custom_loss_function = test_custom_loss_functions

    print("Train data:")
    preview_function(train_data)
    print("Test data:")
    preview_function(test_data)
    print("Test custom losses:")
    custom_loss_function(train_data)

    # Define Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            min_delta=0.01,
            patience=15,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
            start_from_epoch=3,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_sparse_categorical_accuracy",
            factor=0.3,
            patience=5,
            verbose=0,
            mode="auto",
            min_delta=0.02,
            cooldown=0,
            min_lr=1e-6
        ),
        tf.keras.callbacks.BackupAndRestore(
            f"{MODEL_DIR}/checkpoints",
            save_freq="epoch",
            delete_checkpoint=True,
            save_before_preemption=False
        )
    ]

    # train model
    history = autoencoder.fit(
        train_data,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=test_data,
        callbacks=callbacks
    )

    # save
    autoencoder.save(f"{MODEL_DIR}/{dt.datetime.now():%Y%m%d%H%M%S}_autoencoder.tf")
    # autoencoder.save(f"{MODEL_DIR}/{dt.datetime.now():%Y%m%d%H%M%S}_autoencoder.h5")
    # autoencoder.save(f"{MODEL_DIR}/{dt.datetime.now():%Y%m%d%H%M%S}_autoencoder.keras")
