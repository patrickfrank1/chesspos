# Research to validate my ideas

## Research prompt

I want to built an ai model that can analyze and interpret chess positions. It should be able to represent a chess position as a vector semantically to enable search. Also the vector representation can be used to train downstream models that evaluate the position or classify typical patterns. How should I train such an algorithm. Consider that a game is the overall unit of concern. A game is made up of a sequence of positions. Each position represents the current game state at that time with perfect information. However, some patterns only emerge from a sequence of positions, e.g. captures, attacks, strategies, etc. Should I train the embedding algorithm on samples of a position, two adjacent positions, a sequence of positions or on some other unit? What should the loss function be?

## GLM-5 (Z.ai):

### Answer: https://chat.z.ai/s/c9d99892-c798-4901-aefc-d46f77894d4e

Building a chess position embedding model is a fascinating problem because it sits at the intersection of computer vision (the board), sequence modeling (the game), and logic (the rules).

To build a model that generates semantic vectors useful for search and downstream tasks, you should shift your perspective from treating a position as a static snapshot to treating it as a dynamic state within a trajectory.

Here is a detailed breakdown of how to approach the training of such a model.

1. The Unit of Sampling: Sequences (Trajectories)
You asked whether to train on single positions, adjacent pairs, or sequences. The answer is sequences, but processed in a specific way.

Why not single positions?
A single position is "perfect information" regarding piece placement, but it lacks "semantic context."

Example: A position where White has a King on g1 and Rook on f1. Is this a "safe king" or a "back-rank mate threat"? You cannot know without seeing the previous moves (did Black just play ...Qd1?) or understanding the pressures currently being applied.
A single position embeds structure but fails to capture tempo and intent.
Why sequences?
Chess semantics are defined by transitions.

Tactics: A fork or a pin is not a static state; it is a relationship between the current state and the legal moves available.
Strategy: "Space advantage" or "bad bishop" are static features, but "Kingside attack" is a temporal process.
The Recommendation: Sliding Windows
Use a Sliding Window approach. Sample a contiguous sequence of positions (e.g., 8 to 12 half-moves) from a game.

The model processes this sequence to produce an embedding for the current position (the last position in the window).
This allows the model to "watch" the game unfold, understanding the flow (e.g., "White is piling up pieces on the Kingside") before encoding the final state.
2. Model Architecture
To handle a sequence of positions, you should use a Transformer Encoder (similar to BERT in NLP).

Input Representation: Represent each position as a tensor (e.g., 8x8x12 for 12 piece types). A sequence of N positions becomes a sequence of tensors.
Tokenization: You can treat each entire board state as a "token," or (better yet) flatten the board into a sequence of 64 patch tokens and use a Vision Transformer (ViT) approach where temporal attention allows the model to mix information across frames.
Output: The model outputs a vector representation (embedding) for the final position, conditioned on the history observed in the window.
3. The Loss Function: Multi-Task Learning
Since you do not have explicit labels for "semantic similarity," you must use self-supervised and proxy tasks to force the model to learn meaningful representations. You should combine these into a Multi-Task Loss Function.

$$L_total =λ_1 L_next_move +λ_2 L_value +λ_3 L_contrastive$$
​
Task A: Next-Move Prediction (The "Syntax" Loss)
This is the standard "Language Modeling" objective applied to chess.

Objective: Given the sequence of positions $P_t−n …P_t, predict the move M_t+1$.
Why: To predict the next move, the model must understand tactics, threats, and piece coordination. If it cannot predict the next move, it does not understand the position. This ensures the vector encodes tactical viability.
Task B: Value Regression (The "Strategy" Loss)
Objective: Predict the outcome of the game (or the evaluation score) from the current position.
Source: Use Stockfish evaluations or the game result (1-0, 0-1, 1/2-1/2) as labels.
Why: This forces the model to compress concepts like "material advantage," "king safety," and "pawn structure" into the vector. Two positions with equal material but different safety scores will have different vectors because their values differ.
Task C: Contrastive Learning (The "Search" Loss)
This is the most critical component for enabling similarity search.

Objective: Pull similar positions closer in vector space and push dissimilar ones apart.
Defining "Similar": This is tricky in chess.
Transpositions: Positions reached via different move orders are identical. The model should map them to the exact same point (distance≈0).
Strategic Similarity: Positions with the same pawn structure but different piece placements should be close.
Implementation (InfoNCE / Triplet Loss):
Anchor: The current position sequence.
Positive Sample:
Transposition: The same board state reached in a different game (requires a database lookup or buffer).
Future State: The position 5 moves later in the same game (forces the model to learn strategic continuity).
Augmentation: Apply a legal sequence of "null moves" (triangulation) if possible, or slightly perturb the position with a noise-tolerant transformation.
Negative Sample: Random positions from other games.
4. Putting it all together: The Training Pipeline
Data Preparation:
Parse PGN files.
For every position P_t, create a training sample containing the window 
[P_t−7 ,…,P_t].
Calculate Stockfish evaluation for P_t (optional but recommended).
Batch Composition:
Create batches where a single game contributes multiple overlapping windows. This helps the model learn temporal consistency.
Training Loop:
Pass the sequence through the Transformer Encoder.
Extract the [CLS] token or the final positional embedding as the Vector v_t.
Head 1 (Move): Dot product of v_t with embeddings of legal moves (Cross-Entropy Loss).
Head 2 (Value): Feed v_t through an MLP to predict a scalar (MSE Loss vs. Stockfish eval).
Head 3 (Contrastive): Maximize cosine similarity between v_t and the vector for P_t+5 (future state), while minimizing similarity with a random position.
5. Why this works for your Downstream Tasks
By training this way, your vector v_t becomes a powerful semantic representation:

Search: Because of the Contrastive Loss, if you search for a position with an "Isolated Queen's Pawn," the model will return other positions with IQPs, even if the pieces are slightly different, because those positions share similar "Next-Move" patterns and "Value" evaluations.
Pattern Classification: Because of the Next-Move prediction task, the vector inherently encodes tactical patterns. A "pin" pattern creates specific constraints on the next move; the model must encode that constraint in the vector to solve the prediction task.
Evaluation: The Value Regression head ensures the vector smoothly transitions from losing to winning states, making it excellent for training lightweight downstream evaluators.
Summary Recommendation
Train on sequences (sliding windows). This allows the model to "watch" the game state evolve. Use a Transformer Encoder with a Multi-Task Loss combining Next-Move Prediction (tactics), Value Regression (strategy), and Contrastive Learning (similarity).

### Commentary

- Some good ideas I have contemplated before.
- Also interestingly it comes up with the same position encoding and a similar tokenization strategy for CNN and attention!
- I think my tokenization strategy is superior (tokens are pieces, plus special tokens for castling, ep. and turn on an 8 by 8 grid).
- In particular I have contemplated the losses before, but the question is if it's not too complex. Some composite BERT losses were famously harming model performance and only a simple MLM objective survived.
- I was unsure of position vs. sequence, but I guess sequence is better then, if performance allows.
- When challenged GLM favours the InfoNCE loss. I don't like that too much.
- Secondarily it recommends next move prediction. I like that way more, actually should be implemented as next position prediction.

Your Goal --> Recommended Approach
Semantic search / clustering --> Temporal Contrastive Learning (positions from same game = positive pair)
Tactical accuracy --> Pretrain with Next-Move Prediction first, then contrastive fine-tune

## Gemini

### Answer: https://gemini.google.com/share/bac0491eb7c7

This is a fascinating architectural challenge. To build a model that captures both the static state of the board and the dynamic "flow" of the game (strategy, attacks, momentum), you need to move beyond simple snapshot analysis.Here is a breakdown of how to approach training an embedding model for chess that balances static evaluation with sequential context.1. The Unit of Training: Sequence vs. SnapshotYou asked whether to train on single positions, pairs, or sequences.Recommendation: Train on a Window of Moves (Context Window)While a single position contains "perfect information" (all pieces are visible), it lacks "historical intent." A knight on f3 might be there to defend h2, or it might be retreating from an attack. You cannot know the difference without seeing the previous moves.Why not just one position? Static CNNs (like early AlphaZero layers) are great at evaluation, but if you want semantic search (e.g., "find me games with a Kingside attack"), you need the vector to encode the process, not just the state.Why not the whole game? Games vary wildly in length (20 moves to 100+). Encoding an entire game into one fixed vector often dilutes specific tactical moments.The Ideal Unit: A sliding window of positions (e.g., 8–16 half-moves).This allows the model to see the "setup" and the "execution." You can generate an embedding for the center frame of that window, enriched by the context of the frames before and after it.2. Input RepresentationBefore feeding data to the model, you must decide how to represent the board.Bitboards (CNN approach): Represent the board as a stack of $8 \times 8$ planes (e.g., $8 \times 8 \times 12$ for piece types + extra planes for castling rights/en passant). This preserves spatial locality.Token Sequence (Transformer approach): Represent moves as text tokens (e.g., "e4", "Nf3") or squares as tokens.Given your goal is "semantic search" and "patterns," Transformers are currently the state-of-the-art for this. They excel at attention mechanisms, which are perfect for realizing that a Bishop on a1 is exerting force on h8.3. Training Methodologies & Loss FunctionsTo learn a "semantic vector," you should use Self-Supervised Learning (SSL). You don't need labeled data (win/loss) initially; you just need the model to understand the structure of chess.Here are the two best architectures for your goal:A. Contrastive Learning (The "Search" Approach)This is best for your goal of enabling search and finding similar positions.Concept: Use an approach similar to CLIP or SimCLR.The Data: Take a sequence of moves (the "Anchor").Positive Sample: The same sequence slightly perturbed (e.g., change the move order of non-interacting pieces, or mask one move) or the next sequence in the same game.Negative Sample: A random sequence from a different game.The Loss Function: InfoNCE Loss (Noise Contrastive Estimation).$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k_+)/\tau)}{\sum_{i=0}^{K} \exp(\text{sim}(q, k_i)/\tau)}$$This forces the model to push the vectors of "logically connected" chess positions closer together and push unrelated positions apart in vector space.B. Masked Move Modeling (The "BERT" Approach)This is best for understanding strategy and downstream classification.Concept: Treat a game as a sentence and moves as words.The Task: Mask 15% of the moves in the sequence and ask the model to predict the missing move based on the surrounding context.The Loss Function: Cross-Entropy Loss.The model must understand that if the sequence involves a "Greek Gift Sacrifice," the missing move must be $B \times h7$. This forces the internal embeddings to learn the concept of "sacrifice."4. Downstream TasksOnce trained, you freeze the "Encoder" part of your model. This encoder now turns any chess position (plus its context) into a dense vector (e.g., size 512).You can then train lightweight "Heads" on top of these vectors for your specific needs:Evaluation Head: Input: Vector $\rightarrow$ Output: Win Probability (Scalar).Classification Head: Input: Vector $\rightarrow$ Output: Label (e.g., "Fork", "Pin", "Quiet Game").Search: simply uses Cosine Similarity between vectors.Summary of RecommendationFeatureRecommendationInput UnitA sequence of moves (Context Window of ~10-16 moves).ArchitectureTransformer Encoder (BERT-like).Training ObjectiveContrastive Learning (InfoNCE) if search is priority.Masked Modeling (Cross-Entropy) if strategy classification is priority.RepresentationSequence of tokens (moves) or flattened board patches.Would you like me to write a Python code snippet showing how to implement the InfoNCE loss function for this specific chess data structure?

### Commentary:

- nothing new here

## ChatGPT

### Answer: https://chatgpt.com/share/6990bc44-8798-8011-9eed-4f4a594ccc7f

Data and Position Representation
Chess positions are naturally encoded as perfect-information states. In practice one can encode each board as, e.g., a 8×8×12 binary bitboard tensor or a FEN string with piece placements, side-to-move, castling rights, etc. (For example, Hamara et al. tokenize FEN to a fixed-length sequence.) It can help to include a short history of recent moves: many chess networks include the last few half-moves as extra channels (e.g. AlphaZero’s input encodes the last 7 positions). This supplies context for emerging tactics without storing full game history.

Model Architecture for Embedding
Any differentiable model (CNN, Transformer, GNN, etc.) can be used to embed a board state into a D-dimensional vector. For example, ChessLM (Hull 2025) uses a ViT-style transformer on 8×8 inputs and outputs a 256‑dimensional embedding, while Hamara et al. (2025) use a tokenized FEN into a multi-layer transformer encoder. Graph Neural Networks are also promising: by treating squares (and even moves) as nodes/edges, new GNN layers (e.g. “AlphaGateau”) can handle arbitrary board variants and include edge features for moves. Regardless of architecture, the key is that the network’s final output (often via a special [CLS] token or global pooling) is a fixed-size vector representation of the position.

Training Units: Single Positions vs. Pairs vs. Sequences
Single positions. The simplest unit is an isolated board state. One can train the model to encode each position so that its embedding captures structural or evaluative information. For example, Hamara et al. train on millions of individual positions labeled by a strong engine’s win-probability, using a contrastive loss that pulls together positions with similar evaluation and pushes apart those with different scores. ChessLM (Hull 2025) trains on single positions with a “masked piece prediction” task: randomly mask some pieces on the board and train the model to predict them. These single-board tasks teach the model about typical chess configurations and positional features (pawn structures, piece mobility, etc.).

Adjacent pairs. To incorporate move dynamics, one can train on pairs of consecutive or nearby positions. For instance, ChessLM also uses a moves-difference prediction task: given two positions from the same game, predict the number of plies (half-moves) between them. This forces the model to learn something about legal moves and sequence length. DeepChess (David et al. 2017) similarly uses an unsupervised pretraining on single positions, then supervised training on pairs: it learns to compare two positions and predict which is “more favorable”. In general, pairwise tasks (next-position prediction, temporal distance, win/loss ordering) allow the embedding to capture information about how the position can evolve.

Longer sequences. Some patterns (e.g. mating nets, long-term strategies) span multiple moves. One could train on short windows of positions (e.g. 3–7 positions) with sequence models. For example, one could treat a sequence of boards as a time-series and apply contrastive predictive coding (CPC) or sequence-to-sequence objectives: encode positions into latent vectors and train an autoregressive module to predict future embeddings (using a contrastive loss). Alternatively, one might augment the input by including N previous frames (as AlphaZero does), but still output a single embedding of the current state. Training on full game sequences is more complex and data-intensive, but it could help the network “see” a tactic developing. In practice, a common compromise is to use mostly single-position inputs (for indexing/search) but add auxiliary losses or inputs that link adjacent states.

Loss Functions and Objectives
A variety of loss functions can be used, often in combination:

Contrastive (Metric) loss. A popular approach is to use an InfoNCE-style contrastive loss. Here each anchor position’s embedding (z_i) is compared to “positive” and “negative” examples. For example, one can treat an augmented copy of the same position (or a very similar position) as positive, and other random positions as negatives. The loss then minimizes (-\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i,z_k)/\tau)}). Chen et al. (2020) show this encourages embeddings of similar states to be close. When domain labels (like win probabilities or tactical themes) are available, supervised contrastive (SupCon) can be used: multiple positives per anchor are allowed, e.g. all positions with nearly equal engine evaluation. Hamara et al. use exactly this: they label all positions with evaluation within 0.05 (in win-probability) as positives and sample negatives otherwise. In short, contrastive losses (InfoNCE/SupCon) structure the latent space so that distance reflects semantic or evaluative similarity.

Triplet or Ranking loss. Closely related is a triplet loss or margin ranking: select (anchor, positive, negative) such that (d(anchor,positive) + m < d(anchor,negative)). For chess, an anchor could be a board, a positive a slightly stronger position, and a negative a much weaker one. This enforces an embedding geometry aligned with strength. (SupCon is a generalization of this idea.)

Predictive (Reconstruction) loss. Predictive tasks encourage the model to capture information useful for reconstruction. For example, an autoencoder loss would force the network to reconstruct the board from its latent vector. Alternatively, masked piece prediction (as in ChessLM) uses a cross-entropy loss: mask some squares and train the model to correctly identify the missing pieces. This is analogous to BERT’s masked language modeling, adapted to chess. Such losses make the embedding capture local and non-local relationships between pieces. One can also predict moves: e.g. given a position, predict the next move or a sequence of moves (as in a typical move-prediction policy network).

Sequence-prediction loss. If training on sequences, one might use a CPC-style loss: encode each position, then use a predictor to guess embeddings of future positions, using contrastive loss for correctness. This explicitly trains the model to understand temporal dynamics (captures, exchanges, threats).

Supervised labels. If explicit labels are available, include them. For example, predict the engine’s evaluation (regression with MSE) or game outcome (binary cross-entropy). Stockfish-style evaluation can supervise the embedding (as Hamara et al. do via contrastive sampling). ChessLM is not trained to predict evaluation, so its embeddings capture structural similarity rather than precise strength. But one could combine an evaluation/policy head on top of the embedding, training it with standard supervised losses.

Multi-task combination. In practice one often uses a weighted sum of losses. For example, ChessLM combines masked-piece cross-entropy and move-distance prediction. Another approach is to alternate tasks per batch (e.g. some batches for contrastive, some for masked). Balancing these objectives allows the network to learn rich representations: contrastive losses shape the geometry of latent space, while predictive losses ensure details of the board are encoded.

Choosing the Training Unit
Which unit to use depends on trade-offs:

Single-position training is simplest and scales easily. It makes each position’s embedding depend only on that board. This suffices to capture many static patterns (pawn structure, piece activity, tactical motifs that are locally visible). It also aligns well with the goal of semantic search: embeddings reflect position similarity directly. However, purely single-position training may miss long-range or timing-based patterns (e.g. an ongoing discovered attack or a multi-move strategy).

Pairwise training adds awareness of immediate context. By including adjacent positions (or pairs of positions from the same game) in training, the model can learn how moves change the state. For example, a model could be given (position_t, position_{t+1}) and trained to predict the move, or simply trained contrastively so that close-in-time positions are somewhat “close” in embedding. ChessLM’s move-distance task is an example: it uses pairs from actual game play. Using pairs still yields a vector per position (the embedding network can process each position individually or jointly), but the loss ties embeddings of related positions.

Sequence (multi-step) training is richest but also most complex. One could feed a small sequence (e.g. 3–5 boards) into the model (as a 3D input or via a temporal model) and train it on a sequence-level task (predict the next position or an outcome). Another way is to use the sequence to generate multiple contrastive positives/negatives: e.g. treat a future board as a positive sample for the current board (or vice versa). Contrastive predictive coding would use many negatives to force the encoder to capture information predictive of the future. In chess, however, many key patterns are already “encoded” in the static geometry of the current board (captures, pins, sacrifices), so a practical compromise is often to include only a few prior/next frames (as extra input features) rather than whole games. Notably, AlphaZero-based models typically include the last 7 half-moves in input channels. This shows that adding short-term history can help the embedding know, e.g., which pawn just moved (for en-passant) or that a piece is pinned and just moved, without having to encode the entire game.

Loss Function Recommendations
In summary, promising loss choices include:

Contrastive/Siamese losses: Use InfoNCE or triplet losses to enforce that similar positions (by some criterion) have nearby embeddings. For example, treat positions from the same opening or with similar engine eval as positives. If labeled positions (by strength or motif) are available, use supervised contrastive so all positives in a class are pulled together.

Masked/autoencoding losses: Train the network to reconstruct parts of the board, e.g. predict masked pieces (cross-entropy loss). This forces the model to learn the relationships between pieces and fill in missing information, analogous to language models.

Next-step/move prediction: Include a head that predicts the next move or next board (e.g. cross-entropy over legal moves, or regression to piece displacement). This can be trained jointly and encourages the embedding to capture dynamics.

Evaluation/regression loss: If the goal is also strong evaluation or search, add an MSE loss to regress a known evaluation or game outcome. One can even formulate a ranking loss between positions of different outcomes (win vs loss).

Composite loss: In practice, combine multiple objectives. For example, total loss = λ1·ContrastiveLoss + λ2·MaskedPieceLoss + λ3·NextMoveLoss + …, tuning weights λi. ChessLM effectively does two tasks, while HAMARA’s approach is primarily a single contrastive objective. A combined approach would leverage the strengths of each.

Finally, choose an embedding dimension and model capacity large enough: Hamara et al. note that embedding quality (and engine strength) improved with larger model size and higher embedding dimensionality
. In summary, a hybrid training strategy is likely best: use single-position self-supervision for structural features, pairwise/temporal tasks for dynamics, and a contrastive metric loss (with or without supervision) to tie it all together. This yields embeddings that support semantic search (similar positions are close) and can feed downstream evaluators or classifiers trained on top of those embeddings.

Sources: Recent work on chess embeddings and planning uses supervised contrastive losses on single-board encodings, as well as multi-task self-supervised objectives (masked piece prediction, move-distance regression). Both paradigms inform how to train a position-embedding network.

### Commentary

- not much new here either
- but it includes some really interesting work into some very similar research from last year
  - https://bluehood.github.io/research/benh_Beyond_Evaluation__Learning_Contextual_Chess_Position_Representations_2025.pdf
  - https://huggingface.co/odestorm1/chesslm#:~:text=The%20core%20of%20the%20model,represented%20by%20a%20FEN%20string
  - https://github.com/bluehood/Encoder-ChessLM/
- while it's cool that the idea seems to work in general (my previous experiments also suggested that), his model has some limitations
  - purely autoregressive (BERT style) --> decoder model gpt style should work better and also enable prediction
  - he is training on NCE, next move prediction not explored as much
