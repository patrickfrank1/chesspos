# chess-embedding
Embedding based chess position search and embedding learning for chess positions

This repository allows you to search a chess position against billions of chess positions in millions of games and retrieve similar positions. You can also build your own database with the provided tools. Additionally The projects experiments with embeddings learned from bitboards using the triplet neural network architecture. Feel free to try out your own embedding models to improve the embedding based search retrieval.

## Guide

1. Install the package
2. Search your positions in a provided database
3. Extract positions from your own database for search and metric learning
4. Train and evaluate your own chess position embeddings
5. Contribute
6. Cite this project

## Install the package

Make sure you have python3 installed. You will also need the following packages:
- [h5py](https://github.com/h5py/h5py) to read and write large chunks of data
- [python-chess](https://github.com/niklasf/python-chess) for parsing chess games
- [faiss](https://github.com/facebookresearch/faiss) for billion scale nearest neighbor search

and numpy.

All packages except for faiss can be pip installed. To instal faiss either use anaconda, e.g.

```conda install faiss-cpu -c pytorch```

or 
