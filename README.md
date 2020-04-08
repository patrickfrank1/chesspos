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
Congartulations you have successfully installed the package. It contains the following modules:
- `chesspos.utils`: general purpose functions,
- `chesspos.convert`: convert between different chess position encodings like fen, bitboards and chess.Board(),
- `chesspos.pgnextract`: functions to extract and save bitboards from pgn files,
- `chesspos.binary_index`: functions for loading and searching of bitboards in faiss.

Furthermore this repository contains folders for tests, demos, command line tools and data files.

## 2. Demo: Search your positions in a provided database

![animation of demo notebook](./demo/gif/animation.gif)
