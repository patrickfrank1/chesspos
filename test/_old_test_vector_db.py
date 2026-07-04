import pytest
import numpy as np

from src.search.db import MilvusVectorStore
from src.preprocessing.board_representation import boolean_to_byte_vector, byte_to_boolean_vector


def generate_embeddings(num_embeddings, dimensions, embedding_type):
    if embedding_type == "float":
        vector = np.random.random((num_embeddings, dimensions)).tolist()
        return vector, vector
    elif embedding_type == "binary":
        raw_vector = (np.random.random((num_embeddings, dimensions)) > 0.5).astype(int)
        binary_vector = boolean_to_byte_vector(raw_vector)
        return raw_vector, binary_vector





@pytest.fixture(params=["float", "binary"])
def milvus_store(request):
    store = MilvusVectorStore(embedding_dimensions=128, embedding_type=request.param)
    yield store
    store.delete_collection()


def test_milvus_vector_store_insert_and_search(milvus_store: MilvusVectorStore):
    # Insert some example embeddings
    top_k = 5
    dimensions = 128
    example_ids = np.random.randint(1, 100, size=(40))
    example_vectors, example_embeddings = generate_embeddings(40, dimensions, milvus_store.embedding_type)
    milvus_store.insert_embeddings([example_ids[:20], example_embeddings[:20]])
    milvus_store.insert_embeddings([
        {"id": id, "embedding": embedding}
        for id, embedding in zip(example_ids[20:], example_embeddings[20:])
    ])

    # Create index
    milvus_store.create_default_index()

    # Search for similar embeddings
    query_vectors, query_embeddings = generate_embeddings(2, dimensions, milvus_store.embedding_type)
    search_results = milvus_store.search_by_embeddings(query_embeddings=query_embeddings, top_k=top_k)

    # Plausibility check for results
    assert len(search_results) == 2

    result_ids = search_results[0].ids
    assert len(result_ids) == top_k
    assert isinstance(result_ids[0], int)

    result_distances = search_results[0].distances
    assert len(result_distances) == top_k
    assert isinstance(result_distances[0], float)

    result_fields = search_results[0][0].fields
    assert len(result_fields) == 2
    assert isinstance(result_fields, dict)
    assert "id" in result_fields.keys()
    assert "embedding" in result_fields.keys()
    result_vector = result_fields["embedding"]
    if milvus_store.embedding_type == "binary":
        assert isinstance(result_fields["embedding"], bytes)
        result_vector = byte_to_boolean_vector(result_fields["embedding"], original_shape=(1, dimensions))[0]
    result_in_examples = np.all(np.isclose(np.array(example_vectors), np.array(result_vector)), axis=1)  # broadcasting!
    assert np.any(result_in_examples)


def test_milvus_vector_store_search_by_ids(milvus_store: MilvusVectorStore):
    # Insert some example embeddings
    dimensions = 128
    example_ids = np.random.randint(1, 100, size=(4))
    example_vectors, example_embeddings = generate_embeddings(4, dimensions, milvus_store.embedding_type)
    milvus_store.insert_embeddings([example_ids[:2], example_embeddings[:2]])
    milvus_store.insert_embeddings([
        {"id": id, "embedding": embedding}
        for id, embedding in zip(example_ids[2:], example_embeddings[2:])
    ])

    # Create index
    milvus_store.create_default_index()

    # Query ids
    query_results = milvus_store.search_by_ids(query_ids=example_ids[:2])

    # Plausibility check for results
    assert len(query_results) == 2
    for result in query_results:
        assert result["id"] in example_ids
        result_vector = result["embedding"]
        if milvus_store.embedding_type == "binary":
            assert isinstance(result["embedding"][0], bytes)
            result_vector = byte_to_boolean_vector(result["embedding"][0], original_shape=(1, dimensions))[0]
        ground_truth = np.array(example_vectors)[np.argwhere(example_ids == result["id"])]
        assert np.allclose(ground_truth, np.array(result_vector))
