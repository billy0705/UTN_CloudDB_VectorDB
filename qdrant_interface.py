from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchParams, SearchRequest
from qdrant_client.http.models import VectorParams, Distance


class QdrantInterface:
    def __init__(self, host='localhost', port=6333):
        self.host = host
        self.port = port
        self.client = None
        self.connect_server()

    def connect_server(self):
        self.client = QdrantClient(url=f"http://{self.host}:{self.port}")

    def disconnect_server(self):
        self.client = None

    # def create_table(self, collection_name, vector_size):
    #     self.client.recreate_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    #     )


# test
qdrant_interface = QdrantInterface(host='localhost', port=6333)
qdrant_interface.connect_server()
print("Connected.")
qdrant_interface.disconnect_server()
print("Disconnected.")