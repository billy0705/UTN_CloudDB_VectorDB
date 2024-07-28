from qdrant_client import QdrantClient
from qdrant_client.http.models import (VectorParams, Distance,
                                       PointStruct, HnswConfig)
import os
import pandas as pd


class QDrantInterface:
    def __init__(self, data_path):
        self.data_path = data_path
        self.conn = None
        self.connect_server()
        pass

    def connect_server(self):
        self.conn = QdrantClient(path=self.data_path)

    def disconnect_server(self):
        self.conn = self.conn.close()

    def create_table(self, collection_name, vector_size, metric="Cosine",
                     index_types=None):
        if metric == "Cosine":
            dist = Distance.COSINE
        elif metric == "L2":
            dist = Distance.EUCLID
        if index_types is not None:
            index_config = HnswConfig(
                m=16,
                ef_construct=64,
                full_scan_threshold=1000
            )
        self.conn.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size,
                                        distance=dist),
            hnsw_config=index_config
        )

    def drop_table(self, collection_name):
        self.conn.delete_collection(
            collection_name=collection_name
        )

    def _get_directory_size(self, directory):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    def get_size_of_table(self, collection_name):
        qdrant_data_size = self._get_directory_size(
            f'{self.data_path}/collection/{collection_name}')
        # print(f"Size of data in Qdrant: {qdrant_data_size} bytes")
        # print(qdrant_data_size)
        return qdrant_data_size

    def insert_single_vector(self, collection_name, vector):
        self.conn.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=1, vector=vector.tolist())]
        )

    def transfer_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        vectors = df.to_numpy()
        data = [PointStruct(id=i, vector=vector.tolist())
                for i, vector in enumerate(vectors)]
        return data

    def insert_vector_from_csv(self, collection_name, points):
        self.conn.upsert(collection_name=collection_name, points=points)

    def get_rows_cnt(self, collection_name):
        collection_info = self.conn.get_collection(collection_name)
        # print(collection_info)
        # result = json.dumps(collection_info, indent=4)
        return collection_info.points_count

    def similarity_search(self, collection_name, embedding_vector,
                          metric='Cosine', limit=5):
        if metric == "Cosine":
            dist = "Cosine"
        elif metric == "L2":
            dist = "Euclid"
        res = self.conn.search(
            collection_name=collection_name,
            query_vector=embedding_vector,
            limit=limit,
            search_params={"distance": dist}
        )
        result = [{"id": match.id, "score": match.score} for match in res]
        return result[0]["id"], result[0]["score"]
