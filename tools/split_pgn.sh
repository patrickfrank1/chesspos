#! /bin/bash

src="../data/db/lichess_db_standard_rated_2013-01.pgn"
pre="lichess_db_standard_rated_2013-01-"
games=50000

split -l $((18*$games)) -d --additional-suffix=.pgn $src $pre