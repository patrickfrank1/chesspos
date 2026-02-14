import numpy as np

def encodings_to_tensor_triplets(encodings: np.ndarray) -> np.ndarray:
	number_encodings = encodings.shape[0]
	anchor_index = np.random.randint(number_encodings-1)
	positive_index = anchor_index + 1
	negative_index = min((anchor_index + number_encodings // 2), number_encodings)
	triplets = encodings[[anchor_index, positive_index, negative_index],...]
	return triplets.reshape(1, 3, *encodings.shape[1:])