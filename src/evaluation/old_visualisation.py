from typing import Callable

import numpy as np
import chess


def binarize_array(array, threshold=0.5):
    return np.where(array > threshold, True, False)


def print_board(board: chess.Board) -> None:
    board = board.__str__().split("\n")
    output = ""
    output += ''.join([f"{board[i]}\n" for i in range(len(board))]) + '\n'
    print(output)


def _get_sorted_losses(self, sort_fn: Callable, number_samples: int = 1000) -> list[dict]:
    test_generator = self.test_generator.get_generator()
    batch_size = self.test_generator.batch_size
    batches = max(number_samples // batch_size, 1)
    samples = []

    for _ in range(batches):
        x, __ = next(test_generator)
        print(x.shape)
        for j in range(batch_size):
            input = x[j].reshape(1, *x[j].shape)
            loss = self.model.evaluate(input, input, verbose=0)
            samples.append({'input': input, 'loss': loss})

        sort_wrapper = lambda a, b: sort_fn(a['loss'], b['loss'])
        samples.sort(key=cmp_to_key(sort_wrapper))

    return samples


def plot_best_samples(self, number_samples: int)-> None:
    max_loss = lambda x, y: x - y
    best_samples = self._get_sorted_losses(max_loss, number_samples=10*number_samples)[:number_samples]
    examples_out = ""
    print("Best reconstruction examples:")
    for i in range(number_samples):
        best_sample_input = best_samples[i]['input']
        examples_out += str(best_samples[i]['loss']) + '\n\n'
        examples_out += self._compare_input_to_prediction(best_sample_input)
    print(examples_out)


def plot_worst_samples(self, number_samples: int)-> None:
    min_loss = lambda x, y: y - x
    worst_samples = self._get_sorted_losses(min_loss, number_samples=10*number_samples)[:number_samples]
    examples_out = ""
    print("Worst reconstruction examples:")
    for i in range(number_samples):
        worst_sample_input = worst_samples[i]['input']
        examples_out += str(worst_samples[i]['loss']) + '\n'
        examples_out += self._compare_input_to_prediction(worst_sample_input)
    print(examples_out)


def _compare_input_to_prediction(self, input: np.ndarray) -> None:
    prediction = self.model.predict(input)
    prediction_board = self.output_to_board(prediction)
    prediction_board = prediction_board.__str__().split("\n")

    input_board = self.output_to_board(input)
    input_board = input_board.__str__().split("\n")

    for i in range(len(prediction_board)):
        prediction_board[i] = ''.join([
            f'{Fore.RED}{prediction_board[i][j]}{Style.RESET_ALL}'
            if input_board[i][j] != prediction_board[i][j]
            else prediction_board[i][j] for j in range(len(prediction_board[i]))
        ])

    output_str = "Original:          Reconstructed:\n"
    output_str += ''.join([f"{input_board[i]}    {prediction_board[i]}\n" for i in range(len(input_board))]) + '\n'
    return output_str
