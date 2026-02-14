# Documentation of my experiments to find a good embedding for chess positions

## 1. Standard autoencoders as seen on the keras tutorialy page

Description:
- use different standard network architectures
  - dense networks, cnn
  - shallow one layer autoencoders, till up to 10 layer deep autoencoders
- use the standard binary crossentropy loss
- use different chess position encodings
  - 773-dim bitboard vector
  - (8,8,15)-tensor representations
- training on synthetic, legal positions with up to 6 pieces on the board (simulating end-game positions)

Observations:
- loss goes quickly down to a certain threshold, then stays there
- when feeding through the autoencoder, most positions result in an empty board reconstruction

Interpretation:
- these autoencoders suffer from mode collapse
  - model figures out that setting all tensor values is effective, only 6 of ~1000 floats are supposed to be non-zero

## 2. Adressing mode collapse

Description:
- focus on tensor encoding, since that works for dense networks and cnn
- introduce regularization loss terms to prevent mode collapse
  - term to penalize the total number of (non-) zero entries
    - asymptote 1/n as n goes to zero
    - asymptote n as n goes to infinity
  - term to penalize sum of entries per column
    - encourage model to give a probability distribution over pieces for each square
    - square loss if the sum per column deviates from 1.0 if piece or 0.0 if no piece
  - term to penalize the sum of entries per plane
    - encourage the model to conserve the number of pieces per piece type
    - the target value for each plane is given by the number per piece in the input tensor

Observations:
- the loss is significantly higher now
  - goes down slowly, not abruptly
  - the reconstruction error is always greater than one piece, even for 3 piece positions
  - autoencoder outputs are also non-zero now
- what works best?
  - shallow networks perform better that deeper networks
  - cnn don't work well
  - more embedding dimensions work better than less embedding dimensions

Interpretation:
- regularizer terms worked well to prevent mode collapse
- cnn architecture might be wrongly implemented or not suitable for encoding
- what is preventing the model to reconstruct even with no masked fields (perfect information)?
  - since shallow networks perform better, we need to introduce skip connections
  - make models more powerful, with better architectures

## 3. Improve model expressiveness

Description:
- introduce skip connections
- only focus on dense neural networks and tensor representations for now
- try shallow and depper network architectures

Observations:
- loss is much lower now
  - keeps slowly decreasing even with up to 4 masked fileds
  - board reconstructions are usually correct for up to 4 pieces
  - positions with up to 10 pieces are usually missing at least 1 piece

Interpretation:
- skip connections lead to a significant boost in performance even for the rather unsuitable dense networks
- the reconstruction error is still too large for any meaningful application
- what else could be improved?
  - maybe we need a more realistic piece distribution?
  - maybe consider transformers?
  - maybe consider other encodings, positional encodings?

## 4. Improve data set

Description:
- fix the network architecture for now
- extract positions from high quality real world games
  - take lichess games from players > 2400 elo
  - extract a couple positions from each game, not more than 10
  - shuffle positions, so that they are not correlated
  - dataset contains ~10 million games, could be scaled up

Observations:
- results are similar to last experiments
  - no real improvements
  - not good enough for real world applications
  - model could be scaled up, but prohibitive for this set-up
  - problem is simple enough to be solved with the current data set and model size
- even when training without masking, model does not improve performance

Interpretation:
- it's probably not the data, it's the model expressiveness
- there seems no way aroud the transformer

## 5. Implement transformers

Description:
- implement transformer network using keras_nlp library
  - choose encoder, decoder or encoder-decoder architecture
    - 

- encoding choice
- masking
- loss function