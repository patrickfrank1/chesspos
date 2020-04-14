# chess-embedding
Embedding based chess position search and embedding learning for chess positions

This repository allows you to search a chess position against billions of chess positions in millions of games and retrieve similar positions. You can also build your own database with the provided tools. Additionally The projects experiments with embeddings learned from bitboards using the triplet neural network architecture. Feel free to try out your own embedding models to improve the embedding based search retrieval.

## Guide

1. Install the package
2. Demo: Search your positions in a provided database
3. Extract positions from your own database for search and metric learning
4. Train and evaluate your own chess position embeddings
5. Contribute
6. Cite this project

## 1. Install the package

Make sure you have python3 installed. You will also need the following packages:
- [h5py](https://github.com/h5py/h5py) to read and write large chunks of data
- [python-chess](https://github.com/niklasf/python-chess) for parsing chess games
- [faiss](https://github.com/facebookresearch/faiss) for billion scale nearest neighbor search

and numpy.

Additionally for the metric learning part of this project you will need [tensorflow (v2)](https://www.tensorflow.org/).

All packages except for faiss can be pip installed. To install faiss either use anaconda, e.g.

```conda install faiss-cpu -c pytorch```

or follow alternative instructions like [here](https://gist.github.com/korakot/d0a49d7280bd3fb856ae6517bfe8da7a) or [here](https://stackoverflow.com/questions/47967252/installing-faiss-on-google-colaboratory).

Finally pip install this package from source.
```
git clone https://github.com/patrickfrank1/chess-embedding.git
cd chess-embedding
python -m pip install .
# test if installation was successful, the following should run without error
python -c "import chesspos"
```
Congratulations you have successfully installed the package. It contains the following modules:
- `chesspos.utils`: general purpose functions,
- `chesspos.convert`: convert between different chess position encodings like fen, bitboards and chess.Board(),
- `chesspos.pgnextract`: functions to extract and save bitboards from pgn files,
- `chesspos.binary_index`: functions for loading and searching of bitboards in faiss.

Furthermore this repository contains folders for tests, demos, command line tools and data files.

## 2. Demo: Search your positions in a provided database

Now that you installed the package you can ckeck out the demo notebook at [./demo/query_bitboard_db.ipynb](./demo/query_bitboard_db.ipynb).

![animation of demo notebook](https://github.com/patrickfrank1/chess-embedding/demo/gif/animation.gif)

The demo enables you to search a small database of bitbaords for similar positions. I provide some more precompiled databases. The following databases contain high quality games that are generated from [freely available lichess games](https://database.lichess.org/), where we only extracted games with both players above elo 2000 and a time control greater or equal 60+1 seconds.

|          file/link              | positions [million] | download size | RAM needed |
|:-------------------------------:|:-------------------:|:-------------:|:----------:|
| [index_2013.faiss.bz2][1]       |                 1.7 |         12 MB |     171 MB |
| [index_2014.faiss.bz2][2]       |                11.5 |         80 MB |     1.2 GB |
| [index_2015.faiss.bz2][3]       |                  47 |        324 MB |     4.6 GB |
| [index_2016.faiss.bz2][4]       |                 337 |        2.4 GB |      31 GB |
| [index_2020_01_02.faiss.bz2][5] |                 510 |        3.6 GB |      50 GB |

[1]:https://drive.google.com/open?id=1MQKJ6KSmYRyPbIP1ldsNBo-0dGhi-CpQ
[2]:https://drive.google.com/open?id=1eehvnDIbhP4HD6XEH-YeyVJMVX-vRkXc
[3]:https://drive.google.com/open?id=1_abWaGWzkpGd02CYokhWwGlDEEBdCOZl
[4]:https://drive.google.com/open?id=126NbR0EIVzoIU5xd_6eIYPIFz0XrcwEq
[5]:https://drive.google.com/open?id=1u3R5t5jC3I5FFAxywZQ0K4_QZPQLL8cy

However, as you can find out by playing with the notebook the similarity search with bitboards is not optimal, this is why we explore metric learning later on.

## 3. Extract positions from your own database for search and metric learning

The `tools` folder provides useful command line scripts to preprocess pgn files that contain chess positions.

For example, to extract bitboards from all positions of all games in a pgn file open a terminal in the tools foder and run:
```bash
python3 pgn_extract.py ../data/raw/test.pgn --save_position ../data/bitboards/test-bb1.h5
```

This command takes as input the path to you pgn file and wirtes the bitboards to an h5 file at the path specified via `--save_position`. Note: you can drop the .pgn and .h5 file endings and the program will still parse the right files. To ease the file writing process and occupy less ram you can use the `--chunksize` flag, so that your data will be written in chunks, e.g `--chunksize 10000`.

We can also utilize this script to extract tuples of positions for metric learning, to do so run:
```bash
python3 pgn_extract.py ../data/raw/test --save_position ../data/bitboards/test-bb1 --tuples True --save_tuples ../data/train_small/test2-tuples-strong
```
This will extract tuples from each game by virtue of the method `tuple_generator` in `chesspos.pgnextract`. Each generated tuple has the shape (15, 773) and contains a randomly sampled position of each game i *game[i][j]* and randomly sampled positions from the next game as
```
tuple = (game[i][j], game[i][j+1], game[i][j+2], game[i][j+3], game[i][j+4], game[i][(j+14) mod len(game[i])], game[i+1][rand1], ..., game[i+1][rand9])
```

Furthermore the command line script implements two simple filters to subsample  big pgn files. For example
```bash
python pgn_extract.py ../data/raw/test --save_position ../data/bitboards/test-bb2 --chunksize 10000 --tuples True --save_tuples ../data/train_small/test2-tuples-strong --filter elo_min=2400 --filter time_min=61
```
selects only games in which both players have an elo greater or equal to 2400 and where the time control is greater or equal to 61. The time control is calculated as *seconds + seconds per move*, which means a bullet game (60s+0s) is discarded whereas a bullet game with increment (60s+1s) is kept.
