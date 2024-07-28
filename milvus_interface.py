# pip install pymilvus milvus sentence-transformers
# import numpy as np
import pandas as pd
import json
# from milvus import default_server
from pymilvus import MilvusClient, connections
# from time import time


class MilvusInterface:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.connect_server()
        pass

    def connect_server(self):
        self.client = MilvusClient(self.db_path)

    def disconnect_server(self):
        self.client.close()
        pass

    def create_table(self, name, dimention, metric=None, index_types=None):
        if self.client.has_collection(name):
            res = self.client.describe_collection(
                collection_name=name
            )
            print(res)
        else:
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type=metric,
                index_type=index_types,
                index_name="vector_index",
                params={"nlist": 128}
            )

            self.client.create_collection(
                collection_name=name,
                dimension=dimention
            )

            self.client.create_index(
                collection_name=name,
                index_params=index_params
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

    def similarity_search(self, name, embedding_vector, metric=None):
        res = self.client.search(
            collection_name=name,
            data=[embedding_vector.tolist()],
            limit=1,
            search_params={"metric_type": metric, "params": {}}
        )
        result = json.dumps(res, indent=4)
        # print(result)
        return result
