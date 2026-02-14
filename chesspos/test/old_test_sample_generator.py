import numpy as np
from chesspos.utils.decorators import timer
from chesspos.preprocessing.sample_generator import SampleGenerator

def extract_first_sample(samples):
	return samples.reshape((*samples.shape, 1)), samples.reshape((*samples.shape, 1))

@timer
def test_number_samples():
	sample_dir = "chesspos/test/test_data/"
	sample_generator = SampleGenerator(sample_dir, None)
	print(sample_generator.number_samples)

def test_sample_generator():
	sample_dir = "chesspos/test/test_data/"
	sample_generator = SampleGenerator(sample_dir, extract_first_sample, sample_type=np.bool)
	generator = sample_generator.generator_function()
	for _ in range(3):
		print(next(generator)[0].shape)

def test_sample_tf_dataset():
	sample_dir = "chesspos/test/test_data/"
	sample_generator = SampleGenerator(sample_dir, extract_first_sample, sample_type=np.bool)
	tf_dataset = sample_generator.get_tf_dataset()
	print(tf_dataset)
	print(tf_dataset.cardinality().numpy())
	print(list(tf_dataset.take(1)))


if __name__ == "__main__":
	#test_number_samples()
	#test_sample_generator()
	test_sample_tf_dataset()

