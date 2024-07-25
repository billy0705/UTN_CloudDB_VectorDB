from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchParams, SearchRequest
from qdrant_client.http.models import VectorParams, Distance
import os


class QDrantInterface:
    def __init__(self, data_path):
        self.data_path = data_path
        self.conn = None
        self.connect_server()
        pass

    def connect_server(self):
        self.conn = QdrantClient(path=self.data_path)

    def disconnect_server(self):
        self.conn = None

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
        qdrant_data_size = self._get_directory_size(f'{self.data_path}/collection/{collection_name}')
        print(f"Size of data in Qdrant: {qdrant_data_size} bytes")
        print(qdrant_data_size)
        return qdrant_data_size

    def insert_single_vector(self, table_name, vector):
        query = f'INSERT INTO {table_name} (embedding) VALUES (%s)'
        self.conn.execute(query, (vector,))
        self.conn.commit()

    def insert_vector_from_csv(self, table_name, csv_path):
        df = pd.read_csv(csv_path)
        df = df.to_numpy()
        print(f"{df.shape = }")

        for i in range(df.shape[1]):
            vector = df[i, :]
            self.insert_single_vector(table_name, vector)

    def get_rows_cnt(self, table_name):
        query = f'SELECT COUNT(*) FROM {table_name}'
        result = self.conn.execute(query).fetchall()
        # print(result)
        self.conn.commit()
        return result[0][0]
    
    def similarity_search(self, table_name, embedding_vector, metrix):
        if metrix == "l2":
            symbol = "<->"
        elif metrix == "cosine":
            symbol = "<=>"
        else:
            print("Error with metrix type")
            return
        sim_query = f"""
        SELECT id, embedding {symbol} (%s) AS distance
        FROM {table_name}
        ORDER BY distance ASC
        LIMIT 5
        """
        result = self.conn.execute(sim_query, (embedding_vector,)).fetchall()
        # print(result)
        return result
