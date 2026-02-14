import chess

from chesspos.preprocessing.position_processors import (
	board_to_bitboard, bitboard_to_board, board_to_tensor, tensor_to_board
)

start_board = chess.Board(chess.STARTING_FEN)
ep_board = chess.Board("rnbqkbnr/pp1p1ppp/8/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3")
ep_board_state = (chess.WHITE, True, True, True, True, chess.C6)
ep_board_simplified_state = (chess.WHITE, True, True, True, True, None)

def _get_board_state(board):
	return (board.turn, board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK), board.ep_square)

def test_board_to_bitboard():
	board_result = board_to_bitboard(start_board)
	assert board_result.shape == (773,)
	assert board_result.dtype == 'bool'

def test_full_bitboard_conversion():
	ep_bitboard = board_to_bitboard(ep_board)
	reconstructed_board = bitboard_to_board(ep_bitboard)
	print(reconstructed_board.__str__())
	assert ep_board_simplified_state == _get_board_state(reconstructed_board)
	assert ep_board.board_fen() == reconstructed_board.board_fen()

def test_board_to_tensor():
	start_tensor = board_to_tensor(start_board)
	assert start_tensor.shape == (8, 8, 15)
	assert start_tensor.dtype == 'bool'

def test_board_to_tensor_ep():
	ep_tensor = board_to_tensor(ep_board)
	assert ep_tensor[5,2,13] == True

def test_full_tensor_conversion():
	ep_tensor = board_to_tensor(ep_board)
	reconstructed_board = tensor_to_board(ep_tensor)
	print(reconstructed_board.__str__())
	assert ep_board_state == _get_board_state(reconstructed_board)
	assert ep_board.board_fen() == reconstructed_board.board_fen()
