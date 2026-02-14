import keras

from src.search.db import MilvusVectorStore
from src.training.sample_generator import AutoencoderDataGenerator

POSITION_DATA_PATH = "./data/test"
AUTOENCODER_MODEL_PATH = "./model/20231016214738_autoencoder.tf"
BATCH_SIZE = 100
BITBOARD_SIZE = 773
# this should be inferred from the model itself
EMBEDDING_SIZE = 256

class ChessPositionVectorStore:
    def __init__(
        self,
        position_embedding_model: keras.Model | None = None,
        # this will currently provide token sequences or tensors, depending on what is stored in file
        # we might want to index positions directly from pgn, but this requires some refactoring
        position_generator: AutoencoderDataGenerator | None = None,
        bitboard_vector_store: MilvusVectorStore | None = None,
        embedding_vector_store: MilvusVectorStore | None = None
    ):
        self.position_generator = position_generator or get_default_position_generator()
        self.position_encoder = position_embedding_model or get_default_position_embedding_model()
        self.bitboard_vector_store = bitboard_vector_store or get_default_bitboard_vector_store()
        self.embedding_vector_store = embedding_vector_store or get_default_embedding_vector_store()

    def index_positions_from_token_sequence(self):
        for position_batch, _ in self.position_generator:
            embedding_batch = self.position_encoder

def get_default_position_generator() -> AutoencoderDataGenerator:
    return AutoencoderDataGenerator(POSITION_DATA_PATH, batch_size=BATCH_SIZE)

def get_default_position_embedding_model() -> keras.Model:
    autoencoder: keras.Model = keras.models.load_model(AUTOENCODER_MODEL_PATH)
    encoder = autoencoder.get_layer('encoder')
    return encoder

def get_default_bitboard_vector_store() -> MilvusVectorStore:
    return MilvusVectorStore(
        embedding_dimensions=BITBOARD_SIZE,
        embedding_type="binary",
        collection_name="bitboard"
    )

def get_default_embedding_vector_store() -> MilvusVectorStore:
    return MilvusVectorStore(
        embedding_dimensions=EMBEDDING_SIZE,
        embedding_type="float",
        collection_name="embedding"
    )
