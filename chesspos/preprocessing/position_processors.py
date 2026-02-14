import chess
import numpy as np

def board_to_bitboard(board: chess.Board) -> np.ndarray:
	embedding = np.array([], dtype=bool)
	for color in [1, 0]:
		for i in range(1, 7): # P N B R Q K / white
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
	for color in [1, 0]: # white, black
		for i in range(1, 7): # P N B R Q K
			idx = (1-color)*6 + i - 1
			piece = chess.Piece(i,color)
			bitmask = bb[idx*64:(idx+1)*64]
			squares = np.flatnonzero(bitmask)
			for square in squares:
				reconstructed_board.set_piece_at(square,piece)

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

def board_to_tensor(board: chess.Board) -> np.ndarray:
	embedding = np.empty((8,8,15), dtype=bool)
	# one plane per piece
	for color in [1, 0]:
		for i in range(1, 7): # P N B R Q K / white
			index = (1-color)*6 + i - 1
			bmp = np.zeros(shape=(64,)).astype(bool)
			for j in list(board.pieces(i, color)):
				bmp[j] = True
			bmp = bmp.reshape((8,8))
			embedding[:,:, index] = bmp

	# castling rights at plane embedding(:,:,12)
	embedding[0,0,12] = board.has_queenside_castling_rights(chess.WHITE)
	embedding[0,7,12] = board.has_kingside_castling_rights(chess.WHITE)
	embedding[7,0,12] = board.has_queenside_castling_rights(chess.BLACK)
	embedding[7,7,12] = board.has_kingside_castling_rights(chess.BLACK)

	# en passant squares at plane embedding(:,:,13)
	en_passant = np.zeros((64,), dtype=bool)
	if board.has_legal_en_passant():
		en_passant[board.ep_square] = True
	en_passant = en_passant.reshape((8,8))
	embedding[:,:,13] = en_passant

	# turn at plane embedding(:,:,14)
	embedding[0,0,14] = board.turn
	return embedding

def tensor_to_board(tensor: np.ndarray, threshold: float = 0.5) -> chess.Board:
	assert tensor.shape == (8,8,15), f"tensor_to_board encounterer an input with invalid shape {tensor.shape}, expected shape (8,8,15)"
	tensor = np.where(tensor > threshold, 1, 0)

	# set up empty board
	reconstructed_board = chess.Board()
	reconstructed_board.clear()

	# loop over all pieces and squares
	for color in [1, 0]: # white, black
		for i in range(1, 7): # P N B R Q K
			idx = (1-color)*6 + i - 1
			piece = chess.Piece(i,color)
			square_bitmask = tensor[:,:,idx].reshape((64,))
			squares = np.flatnonzero(square_bitmask)
			for square in squares:
				reconstructed_board.set_piece_at(square,piece)
	
	# set castling rights
	castling_rights = ''
	if tensor[0,0,12]: castling_rights += 'Q'
	if tensor[0,7,12]: castling_rights += 'K'
	if tensor[7,0,12]: castling_rights += 'q'
	if tensor[7,7,12]: castling_rights += 'k'
	reconstructed_board.set_castling_fen(castling_rights)

	# set en passant square
	en_passant = tensor[:,:,13].reshape((64,))
	if np.any(en_passant):
		reconstructed_board.ep_square = np.flatnonzero(en_passant)[0]

	# set turn
	reconstructed_board.turn = tensor[0,0,14]
	return reconstructed_board
