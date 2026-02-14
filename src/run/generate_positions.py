import numpy as np

from src.preprocessing.board_representation import board_to_tensor, board_to_token_sequence
from src.preprocessing.generate_board import generate_random_board
from src.preprocessing.extract_board import extract_board
from src.utils.data_loader import save_train_test


def main():
    NUMBER_FILES = 100
    POSITIONS = 100_000
    TRAIN_RATIO = 0.9
    PATH = "./data"
    MODE = "real"  # real | artificial
    ENCODING = "token_sequence"  # token_sequence | tensor

    if MODE == "real":
        extractor = extract_board(f"{PATH}/raw")

    for j in range(NUMBER_FILES):
        print(f"Generating file {j} of {NUMBER_FILES}")
        if ENCODING == "tensor":
            board_encodings = np.empty((POSITIONS, 8, 8, 15), dtype=bool)
        elif ENCODING == "token_sequence":
            board_encodings = np.empty((POSITIONS, 69), dtype=np.int8)

        for i in range(POSITIONS):
            print(f"Generating position {i+1} of {POSITIONS}", end="\r")
            if MODE == "artificial":
                new_board = generate_random_board(max_pieces=6)
            elif MODE == "real":
                new_board = next(extractor)
            if ENCODING == "tensor":
                new_board_encoding = board_to_tensor(new_board)
            elif ENCODING == "token_sequence":
                new_board_encoding = board_to_token_sequence(new_board)
            board_encodings[i] = new_board_encoding

        train_cutoff = int(POSITIONS*TRAIN_RATIO)
        save_train_test(
            path=PATH,
            train_split=board_encodings[:train_cutoff, :],
            test_split=board_encodings[train_cutoff:, :]
        )


if __name__ == "__main__":
    main()
