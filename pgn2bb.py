import chess
import chess.pgn
import sys

from chess import pgn

if len(sys.argv) != 2:
	print("Usage: " + sys.argv[0] + " <PGN file>")
	sys.exit(1)
else:
    pgn = open(sys.argv[1])


game1 = chess.pgn.read_game(pgn)

print(game1)

game2 = chess.pgn.read_game(pgn)

print(game2)


  #for i in range(1,6):
  #    print(board.pieces(i,0))
  #    print("\n vars",(i,0),"\n")
  #    print(board.pieces(i,1))
  #    print("\n vars",(i,1),"\n")

  #for move in game.mainline_moves():
  #    board.push(move)
  #    print(board.fen())
