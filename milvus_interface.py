# pip install pymilvus milvus sentence-transformers
# import numpy as np
import pandas as pd
# from milvus import default_server
from pymilvus import MilvusClient, connections, utility, FieldSchema, CollectionSchema, DataType, Collection, db
# from time import time


class MilvusInterface:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.connect_server()
        pass

    def connect_server(self):
        # self.conn = psycopg.connect(
        #     f"dbname={self.dbname} user={self.user} password={self.password}"
        # )
        # self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        # register_vector(self.conn)
        # conn = connections.connect(host="127.0.0.1", port=19530)
        # database = db.create_database("my_database")
        self.client = MilvusClient(self.db_path)

    def disconnect_server(self):
        self.client.close()
        pass

    def execute_query(self, query):
        # result = self.conn.execute(query).fetchall()
        # self.conn.commit()
        # return result
        pass

    def create_table(self, name, dimention, metrix=None, index_types=None):
        # query = f'''CREATE TABLE IF NOT EXISTS {table_name}
        #  (id bigserial PRIMARY KEY, embedding vector({vector_size}))'''
        # self.conn.execute(query)
        # self.conn.commit()
        if self.client.has_collection(name):
            res = self.client.describe_collection(
                collection_name=name
            )
            print(res)
        else:
            self.client.create_collection(
                collection_name=name,
                dimension=dimention
            )
        pass

    def drop_table(self, name):
        self.client.drop_collection(
            collection_name=name
        )
        pass

    def get_size_of_table(self, name):
        # query = f"""SELECT
        #  pg_size_pretty( pg_total_relation_size('{table_name}'))"""
        # result = self.conn.execute(query).fetchall()
        # self.conn.commit()
        # print(result)
        # return result[0][0]
        return 0
        pass

    def insert_single_vector(self, table_name, vector):
        # query = f'INSERT INTO {table_name} (embedding) VALUES (%s)'
        # self.conn.execute(query, (vector,))
        # self.conn.commit()
        pass

    def transfer_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.to_numpy()
        # print(f"{df.shape = }")
        data = [
            {"id": 0, "vector": df[i, :]}
            for i in range(df.shape[1])
        ]
        return data

    def insert_vector_from_csv(self, name, data):
        _ = self.client.insert(
            collection_name=name,
            data=data
        )
        pass

    def get_rows_cnt(self, name):
        res = self.client.get_collection_stats(
            collection_name=name
        )
        return res['row_count']
        pass

    def similarity_search(self, table_name, embedding_vector, metrix):
        # if metrix == "l2":
        #     symbol = "<->"
        # elif metrix == "cosine":
        #     symbol = "<=>"
        # else:
        #     print("Error with metrix type")
        #     return
        # sim_query = f"""
        # SELECT id, embedding {symbol} (%s) AS distance
        # FROM {table_name}
        # ORDER BY distance ASC
        # LIMIT 1
        # """
        # result = self.conn.execute(sim_query, (embedding_vector,)).fetchall()
        # print(result)
        # return result
