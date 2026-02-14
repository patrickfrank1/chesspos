import tensorflow as tf
from keras import backend as K

DTYPE = 'bfloat16'


@tf.keras.utils.register_keras_serializable()
def sum_squared_loss(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
    y_true = K.cast(y_true, dtype=DTYPE)
    y_pred = K.cast(y_pred, dtype=DTYPE)
    squared_difference = K.square(y_true - y_pred)
    loss = K.sum(squared_difference) / batch_size
    return loss


@tf.keras.utils.register_keras_serializable()
def num_pc_reg(y_true, y_pred):
    epsilon = 1.e-3
    batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
    y_true = K.cast(y_true, dtype=DTYPE)
    y_pred = K.cast(y_pred, dtype=DTYPE)
    pieces_true = K.sum(y_true)
    pieces_predicted = K.sum(y_pred)
    loss = K.square(pieces_true - pieces_predicted) / (epsilon + pieces_predicted) / batch_size
    return loss


@tf.keras.utils.register_keras_serializable()
def pc_column_reg(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
    y_true = K.cast(y_true, dtype=DTYPE)
    y_pred = K.cast(y_pred, dtype=DTYPE)
    piece_representation_true = y_true[:, :, :, :12]
    piece_representation_pred = y_pred[:, :, :, :12]
    sum_over_pieces_true = K.sum(piece_representation_true, axis=3)
    sum_over_pieces_pred = K.sum(piece_representation_pred, axis=3)
    deviation_from_legal = K.square(sum_over_pieces_true - sum_over_pieces_pred)
    loss = K.sum(deviation_from_legal) / batch_size
    return loss


@tf.keras.utils.register_keras_serializable()
def pc_plane_reg(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
    y_true = K.cast(y_true, dtype=DTYPE)
    y_pred = K.cast(y_pred, dtype=DTYPE)
    piece_representation_true = y_true[:, :, :, :12]
    piece_representation_pred = y_pred[:, :, :, :12]
    sum_over_planes_true = K.sum(K.sum(piece_representation_true, axis=2), axis=1)
    sum_over_planes_pred = K.sum(K.sum(piece_representation_pred, axis=2), axis=1)
    deviation_from_legal = K.square(sum_over_planes_true - sum_over_planes_pred)
    loss = K.sum(deviation_from_legal) / batch_size
    return loss


@tf.keras.utils.register_keras_serializable()
def custom_regularized_loss(y_true, y_pred):
    alpha = 1.0
    beta = 0.1
    gamma = 0.1
    loss = sum_squared_loss(y_true, y_pred)
    regularizer_1 = num_pc_reg(y_true, y_pred)
    regularizer_2 = pc_column_reg(y_true, y_pred)
    regularizer_3 = pc_plane_reg(y_true, y_pred)
    return loss + alpha * regularizer_1 + beta * regularizer_2 + gamma * regularizer_3
