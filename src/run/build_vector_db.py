import datetime as dt
from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from pymilvus import connections, utility, Collection

from src.evaluation.visualisation import print_board
from src.preprocessing.board_representation import token_sequence_to_board
from src.search.schema import token_schema, embedding_schema, get_index_definition
from src.training.sample_generator import AutoencoderDataGenerator

@dataclass
class ChessPositionVectorStore:
	encoder_model_path: str = "./model/20231016214738_autoencoder.tf"
	encoding: str = "token_sequence" # token_sequence | embedding
	embedding_key: str = "embedding"
	embedding_dimension: int = 256
	batch_size: int = 16
	collection_prefix: str = "lichess_elite"
	reindex_existing: bool = False
	collection: Collection | None = None

	def __post_init__(self):
		connections.connect("default", host="localhost", port="19530")
		self.collection = self._init_collection()

	def _init_collection(self):
		"""Get a collection or create one if none exists."""
	
	def build_index(self, definition: str) -> None:
		index = get_index_definition(definition)
		self.collection.create_index("embedding", self.embedding_key)

	
if __name__ == "__main__":
	DATA_DIR = "./data/test"
	ENCODER_MODEL_PATH = "./model/20231016214738_autoencoder.tf"
	ENCODING = "token_sequence" # token_sequence | tensor
	EMBEDDING_DIMENSION = 256
	BATCH_SIZE = 16
	COLLECTION = f"lichess_elite"
	REUSE_EXISTING = True
	
	# Set up db connection
	connections.connect("default", host="localhost", port="19530")
	schema = {
		"token": token_schema(dimensions=69),
		"embedding": embedding_schema(dimensions=EMBEDDING_DIMENSION)
	}
	collection = {
		"token": None,
		"embedding": None
	}
	
	for vector in collection.keys():
		if utility.has_collection(f"{COLLECTION}_{vector}"):
			if REUSE_EXISTING:
				collection[vector] = Collection(f"{COLLECTION}_{vector}")  
				print("Warning: Existing collection will be used.")
			else:
				raise ValueError("Collection already exists")
		else:
			collection[vector] = Collection(f"{COLLECTION}_{vector}", schema[vector], consistency_level="Strong")

	# Get data and encoder
	test_data = AutoencoderDataGenerator(DATA_DIR, batch_size=BATCH_SIZE)
	autoencoder: keras.Model = keras.models.load_model(ENCODER_MODEL_PATH)
	encoder: keras.Model = autoencoder.get_layer('encoder')

	# Create embeddings and save them to the database
	i = 0
	for i, batch in enumerate(test_data):
		if i == 1:
			test_tokens = batch[0][10].reshape((1,69)).astype(np.float32)
			break
		i += 1
		# seed_index = int(f"{dt.datetime.now():%Y%m%d%H%M%S}{i}")
		# batch_indices = [seed_index+j for j in range(BATCH_SIZE)]
		# tokens = batch[0].astype(np.float32)
		# embeddings = encoder.predict(tokens)
		# collection["token"].insert([batch_indices,tokens])
		# collection["embedding"].insert([batch_indices,embeddings])
		# collection["token"].flush()
		# collection["embedding"].flush()
		# print(f"\nNumber of entities in Milvus: {collection['embedding'].num_entities}")



	collection["token"].create_index("token", index)
	collection["embedding"].create_index("embedding", index)

	# test
	collection["token"].load()
	collection["embedding"].load()
	test_embedding = embeddings = encoder.predict([test_tokens])
	search_params = {
		"metric_type": "L2",
		"params": {"nprobe": 10},
	}

	result = collection["embedding"].search(test_embedding, "embedding", search_params, limit=5, output_fields=["pk"])

	print("Probe:")
	print_board(token_sequence_to_board(test_tokens[0]))
	for hits in result:
		for hit in hits:
			print(f"hit {hit}")
			res = collection["token"].query(f"pk == {hit.id}", output_fields=["token"])
			tok = np.array(res[0]["token"])
			print_board(token_sequence_to_board(tok))
