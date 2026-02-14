from datetime import datetime

from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, SearchResult
from pymilvus import utility


class MilvusVectorStore:
    DTYPE_PARAMETERS = {
        "float": {
            "dtype": DataType.FLOAT_VECTOR,
            "index_type": "IVF_FLAT",
            "metric_type": "L2"
        },
        "binary": {
            "dtype": DataType.BINARY_VECTOR,
            "index_type": "BIN_IVF_FLAT",
            "metric_type": "HAMMING"
        }
    }
    ID_FIELD = "id"
    EMBEDDING_FIELD = "embedding"

    def __init__(
        self,
        embedding_dimensions: int,
        embedding_type: str,  # float | binary
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = 'example_collection'
    ):
        self.embedding_dimensions = embedding_dimensions
        self.embedding_type = embedding_type
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._connect()
        self.delete_collection()
        self._create_collection_if_not_exists()

    def _connect(self):
        """Connect to the Milvus server."""
        connections.connect(alias='default', host=self.host, port=self.port)
        print("Connected to Milvus")

    def _create_collection_if_not_exists(self):
        """Create a collection if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            self.collection = Collection(name=self.collection_name)
        else:
            if self.embedding_type not in self.DTYPE_PARAMETERS.keys():
                raise ValueError(f"Invalid embedding data type for collection {self.collection_name}.")
            fields = [
                FieldSchema(name=self.ID_FIELD, dtype=DataType.INT64, is_primary=True),
                FieldSchema(
                    name=self.EMBEDDING_FIELD,
                    dtype=self.DTYPE_PARAMETERS[self.embedding_type]["dtype"],
                    dim=self.embedding_dimensions
                )
            ]
            description = f"Collection {self.collection_name} created {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}"
            schema = CollectionSchema(fields, description=description)
            self.collection = Collection(name=self.collection_name, schema=schema)
            print(f"Collection '{self.collection_name}' created.")

    def delete_collection(self):
        """Delete a collection in Milvus."""
        if utility.has_collection(self.collection_name):
            collection = Collection(name=self.collection_name)
            collection.drop()
            print(f"Collection '{self.collection_name}' deleted.")
        else:
            print(f"Collection '{self.collection_name}' does not exist.")

    def create_index(self, params: dict):
        """Create an index on a specified field."""
        index_name = f"{self.EMBEDDING_FIELD}_{params['index_type']}_{params['metric_type']}"
        if 'params' in params.keys() and 'nlists' in params['params'].keys():
            index_name += f"_{params['params']['nlists']}"
        self.collection.create_index(
            index_name=index_name,
            field_name=self.EMBEDDING_FIELD,
            index_params=params
        )
        print(f"Index {index_name} on {self.EMBEDDING_FIELD} created.")
        self.collection.load()

    def create_default_index(self):
        params = {
            "metric_type": self.DTYPE_PARAMETERS[self.embedding_type]["metric_type"],
            "index_type": self.DTYPE_PARAMETERS[self.embedding_type]["index_type"],
            "params": {"nlist": 1024}
        }
        self.create_index(params)

    def insert_embeddings(self, data: list | dict) -> list[str]:
        """Insert embeddings into the collection."""
        if not self.collection:
            raise ValueError("Collection not created or loaded.")
        mr = self.collection.insert(data)
        # print(f"Inserted {len(mr.primary_keys)} embeddings.")
        return mr.primary_keys

    def search_by_embeddings(
        self,
        query_embeddings: list[list[float]] | list[bytes],
        top_k: int = 10,
        nprobe: int = 10
    ) -> SearchResult:
        """Search the collection for similar embeddings."""
        search_params = {
            "metric_type": self.DTYPE_PARAMETERS[self.embedding_type]["metric_type"],
            "offset": 0,
            "params": {"nprobe": nprobe}
        }
        if not self.collection:
            raise ValueError("Collection not created or loaded.")
        search_results = self.collection.search(
            data=query_embeddings,
            anns_field=self.EMBEDDING_FIELD,
            output_fields=[self.ID_FIELD, self.EMBEDDING_FIELD],
            param=search_params,
            limit=top_k,
            expr=None
        )
        return search_results

    def search_by_ids(self, query_ids: list[int]):
        """Search the collection for given ids."""
        if not self.collection:
            raise ValueError("Collection not created or loaded.")
        search_results = self.collection.query(
            expr=f"{self.ID_FIELD} in [{','.join(str(id) for id in query_ids)}]",
            offset=0,
            limit=len(query_ids),
            output_fields=[self.ID_FIELD, self.EMBEDDING_FIELD]
        )
        return search_results
