import chess
import chess.pgn

def pgn2bb(file):

	gamelist = []

	with open(file,'r') as f:
		while True:
			game = chess.pgn.read_game(f)

			if game is None:
				break  # end of file
			else:
				gamelist.append(game)
	
	
	return gamelist

if __name__ == "__main__":
	
	games = pgn2bb(file = "data/test.pgn")
	print(len(games))
	print(games)

#for i in range(1,6):
#    print(board.pieces(i,0))
#    print("\n vars",(i,0),"\n")
#    print(board.pieces(i,1))
#    print("\n vars",(i,1),"\n")

#for move in game.mainline_moves():
#    board.push(move)
#    print(board.fen())
