import chess
import numpy as np

def board_to_bitboard(board):
	embedding = np.array([], dtype=bool)
	for color in [1, 0]:
		for i in range(1, 7): # P N B R Q K / white
			bmp = np.zeros(shape=(64,)).astype(bool)
			for j in list(board.pieces(i, color)):
				bmp[j] = True
			embedding = np.concatenate((embedding, bmp))
	additional = np.array([
		bool(board.turn),
		bool(board.castling_rights & chess.BB_A1),
		bool(board.castling_rights & chess.BB_H1),
		bool(board.castling_rights & chess.BB_A8),
		bool(board.castling_rights & chess.BB_H8)
	])
	embedding = np.concatenate((embedding, additional))
	return embedding

def bitboard_to_board(bb):
	# set up empty board
	reconstructed_board = chess.Board()
	reconstructed_board.clear()
	# loop over all pieces and squares
	for color in [1, 0]: # white, black
		for i in range(1, 7): # P N B R Q K
			idx = (1-color)*6+i-1
			piece = chess.Piece(i,color)

			bitmask = bb[idx*64:(idx+1)*64]
			squares = np.argwhere(bitmask)
			squares = [square for sublist in squares for square in sublist] # flatten list of lists

			for square in squares:
				reconstructed_board.set_piece_at(square,piece)
	# set global board information
	reconstructed_board.turn = bb[768]

	castling_rights = ''
	if bb[770]: # castling_h1
		castling_rights += 'K'
	if bb[769]: # castling_a1
		castling_rights += 'Q'
	if bb[772]: # castling_h8
		castling_rights += 'k'
	if bb[771]: # castling_a8
		castling_rights += 'q'
	reconstructed_board.set_castling_fen(castling_rights)

	return reconstructed_board
