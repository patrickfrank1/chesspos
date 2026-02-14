# %%
import pathlib
import chesspos.preprocessing.pgn_extractor as pe
import chesspos.preprocessing.game_filters as gf
import chesspos.preprocessing.game_processors as gp
import chesspos.preprocessing.position_filters as pf
import chesspos.preprocessing.position_processors as pp
import chesspos.custom_types as ct

# %%
game_processor = gp.GameProcessor(
    is_process_position=pf.filter_piece_count(min_pieces=2, max_pieces=4),
    position_processor=pp.board_to_bitboard,
)
game_processor


# %%
pgn_extractor = pe.PgnExtractor(
    is_process_game = gf.no_filter,
    game_processor = game_processor,
    pgn_path = '/home/pafrank/coding/chess-embedding-learning/data/pgn/lichess_db_standard_rated_2013-01.pgn',
    save_path = '/home/pafrank/coding/chess-embedding-learning/data/processed_small'
)
pgn_extractor

# %%
pgn_extractor._encoding_shape, pgn_extractor._encoding_type, pgn_extractor._game_counter

from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

pgn_extractor.extract(10_000)

profiler.stop()
profiler.print()