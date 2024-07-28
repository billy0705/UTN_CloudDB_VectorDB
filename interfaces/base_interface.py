class QDrantInterface:
    def __init__(self, ):
        pass

    def connect_server(self):
        pass

    def disconnect_server(self):
        pass

    def create_table(self, collection_name, vector_size, metric="",
                     index_types=None):
        pass

    def drop_table(self, collection_name):
        pass

    def get_size_of_table(self, collection_name):
        pass

    def insert_single_vector(self, collection_name, vector):
        pass

    def transfer_csv(self, csv_path):
        pass

    def insert_vector_from_csv(self, collection_name, points):
        pass

    def get_rows_cnt(self, collection_name):
        pass

    def similarity_search(self, collection_name, embedding_vector,
                          metric='Cosine', limit=5):
        pass
