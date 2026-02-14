from __future__ import annotations

import chess
import numpy as np

PIECE_ENCODING = {
    "empty": 15,
    "mask": 16,
    "P": 17,
    "N": 18,
    "B": 19,
    "R": 20,
    "Q": 21,
    "K": 22,
    "p": 23,
    "n": 24,
    "b": 25,
    "r": 26,
    "q": 27,
    "k": 28,
    "turn_white": 29,
    "turn_black": 30,
    "no_castling": 31,
    "castling": 32
}

INVERSE_PIECE_ENCODING = {v: k for k, v in PIECE_ENCODING.items()}


def board_to_bitboard(board: chess.Board) -> np.ndarray:
    embedding = np.array([], dtype=bool)
    for color in [1, 0]:
        for i in range(1, 7):  # P N B R Q K / white
            bmp = np.zeros(shape=(64,)).astype(bool)
            for j in list(board.pieces(i, color)):
                bmp[j] = True
            embedding = np.concatenate((embedding, bmp))
    additional = np.array([
        bool(board.turn),
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ])
    embedding = np.concatenate((embedding, additional))
    return embedding


def bitboard_to_board(bb: np.ndarray) -> chess.Board:
    assert bb.shape == (773,)

    # set up empty board
    reconstructed_board = chess.Board()
    reconstructed_board.clear()
    # loop over all pieces and squares
    for color in [1, 0]:  # white, black
        for i in range(1, 7):  # P N B R Q K
            idx = (1-color)*6 + i - 1
            piece = chess.Piece(i, color)
            bitmask = bb[idx*64:(idx+1)*64]
            squares = np.flatnonzero(bitmask)
            for square in squares:
                reconstructed_board.set_piece_at(square, piece)

    # set turn
    reconstructed_board.turn = bb[768]

    # set castling rights
    castling_rights = ''
    if bb[769]: castling_rights += 'Q'
    if bb[770]: castling_rights += 'K'
    if bb[771]: castling_rights += 'q'
    if bb[772]: castling_rights += 'k'
    reconstructed_board.set_castling_fen(castling_rights)
    return reconstructed_board


def board_to_token_sequence(board: chess.Board) -> np.ndarray:
    pieces = [board.piece_at(square) for square in chess.SQUARES]
    encoding = np.zeros((64+5), dtype=np.int8)

    for i, piece in enumerate(pieces):
        encoding[i] = PIECE_ENCODING[piece.symbol() if piece is not None else "empty"]
    encoding[64] = PIECE_ENCODING["turn_white" if board.turn else "turn_black"]
    encoding[65] = PIECE_ENCODING["castling" if board.has_kingside_castling_rights(chess.WHITE) else "no_castling"]
    encoding[66] = PIECE_ENCODING["castling" if board.has_queenside_castling_rights(chess.WHITE) else "no_castling"]
    encoding[67] = PIECE_ENCODING["castling" if board.has_kingside_castling_rights(chess.BLACK) else "no_castling"]
    encoding[68] = PIECE_ENCODING["castling" if board.has_queenside_castling_rights(chess.BLACK) else "no_castling"]
    return encoding


def _piece_id_to_piece(id: int) -> chess.Piece | None:
    if 17 <= id <= 28:
        return chess.Piece.from_symbol(INVERSE_PIECE_ENCODING[id])
    return None


def token_sequence_to_board(sequence: np.ndarray) -> chess.Board:
    assert sequence.shape == (69,)
    board = chess.Board()
    board.clear()

    piece_map = {
        i: _piece_id_to_piece(piece_id)
        for i, piece_id in enumerate(sequence[:64])
        if _piece_id_to_piece(piece_id) is not None
    }
    board.set_piece_map(piece_map)

    board.turn = sequence[64] == 29
    castling_rights = ""
    if sequence[65] == 32: castling_rights += "K"
    if sequence[66] == 32: castling_rights += "Q"
    if sequence[67] == 32: castling_rights += "k"
    if sequence[68] == 32: castling_rights += "q"
    board.set_castling_fen(castling_rights)
    return board


def board_to_tensor(board: chess.Board) -> np.ndarray:
    embedding = np.zeros((8, 8, 15), dtype=bool)
    # one plane per piece
    for color in [1, 0]:
        for i in range(1, 7):  # P N B R Q K / white
            index = (1-color)*6 + i - 1
            bmp = np.zeros(shape=(64,)).astype(bool)
            for j in list(board.pieces(i, color)):
                bmp[j] = True
            bmp = bmp.reshape((8, 8))
            embedding[:, :, index] = bmp

    # castling rights at plane embedding(:,:,12)
    embedding[0, 0, 12] = board.has_queenside_castling_rights(chess.WHITE)
    embedding[0, 7, 12] = board.has_kingside_castling_rights(chess.WHITE)
    embedding[7, 0, 12] = board.has_queenside_castling_rights(chess.BLACK)
    embedding[7, 7, 12] = board.has_kingside_castling_rights(chess.BLACK)

    # en passant squares at plane embedding(:,:,13)
    en_passant = np.zeros((64,), dtype=bool)
    if board.has_legal_en_passant():
        en_passant[board.ep_square] = True
    en_passant = en_passant.reshape((8, 8))
    embedding[:, :, 13] = en_passant

    # turn at plane embedding(:,:,14)
    embedding[0, 0, 14] = board.turn
    return embedding


def tensor_to_board(tensor: np.ndarray, threshold: float = 0.5) -> chess.Board:
    assert tensor.shape == (8, 8, 15), (
        f"tensor_to_board encounterer an input with invalid shape {tensor.shape}, " +
        "expected shape (8,8,15)"
    )
    tensor = np.where(tensor > threshold, 1, 0)

    # set up empty board
    reconstructed_board = chess.Board()
    reconstructed_board.clear()

    # loop over all pieces and squares
    for color in [1, 0]:  # white, black
        for i in range(1, 7):  # P N B R Q K
            idx = (1-color)*6 + i - 1
            piece = chess.Piece(i, color)
            square_bitmask = tensor[:, :, idx].reshape((64,))
            squares = np.flatnonzero(square_bitmask)
            for square in squares:
                reconstructed_board.set_piece_at(square, piece)

    # set castling rights
    castling_rights = ''
    if tensor[0, 0, 12]: castling_rights += 'Q'
    if tensor[0, 7, 12]: castling_rights += 'K'
    if tensor[7, 0, 12]: castling_rights += 'q'
    if tensor[7, 7, 12]: castling_rights += 'k'
    reconstructed_board.set_castling_fen(castling_rights)

    # set en passant square
    en_passant = tensor[:, :, 13].reshape((64,))
    if np.any(en_passant):
        reconstructed_board.ep_square = np.flatnonzero(en_passant)[0]

    # set turn
    reconstructed_board.turn = tensor[0, 0, 14]
    return reconstructed_board


def boolean_to_byte_vector(boolean_vector: np.ndarray) -> bytes:
    assert boolean_vector.dtype in [bool, int]
    uint8_packed_vector = np.packbits(boolean_vector, axis=-1)
    binary_vector = [bytes(vector.tolist()) for vector in uint8_packed_vector]
    return binary_vector


def byte_to_boolean_vector(byte_vector: list[bytes], original_shape: Tuple[int, int]):
    unpacked_array = np.frombuffer(byte_vector, dtype=np.uint8)
    unpacked_bits = np.unpackbits(unpacked_array)
    unpacked_bits = unpacked_bits[:np.prod(original_shape)].reshape(original_shape)
    return unpacked_bits
