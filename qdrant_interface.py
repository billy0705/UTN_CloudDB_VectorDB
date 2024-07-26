from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchParams, SearchRequest
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import os
import pandas as pd
import numpy as np


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

    def create_table(self, collection_name, vector_size, metrix=None):
        self.conn.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size,
                                        distance=Distance.COSINE),
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
        print(f"Size of data in Qdrant: {qdrant_data_size} bytes")
        print(qdrant_data_size)
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
        return collection_info.vectors_count


# testing
qdrant_interface = QDrantInterface(data_path='./qdrant_data')
print("Connected to Qdrant server.")

qdrant_interface.drop_table('vector_collection')

qdrant_interface.create_table(collection_name='vector_collection', vector_size=1000)
print("Collection created successfully.")

# # Insert a single vector
# try:
#     sample_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
#     qdrant_interface.insert_single_vector(collection_name='vector_collection', vector=sample_vector)
#     print("Vector inserted successfully.")
# except Exception as e:
#     print(f"An error occurred while inserting the vector: {e}")

# # Insert vectors from CSV
# try:
#     qdrant_interface.insert_vector_from_csv(collection_name='vector_collection', csv_path='./data/clustered_vectors.csv')
#     print("Vectors inserted successfully from CSV.")
# except Exception as e:
#     print(f"An error occurred while inserting vectors from CSV: {e}")

# Get row count
try:
    rows_count = qdrant_interface.get_rows_cnt(collection_name='vector_collection')
    print(f"Number of vectors in the collection: {rows_count}")
except Exception as e:
    print(f"An error occurred while getting the row count: {e}")

qdrant_interface.disconnect_server()
print("Disconnected from Qdrant server.")
