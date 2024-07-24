# pip install pymilvus milvus sentence-transformers
# import numpy as np
import pandas as pd
from milvus import default_server
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, db
from time import time

class MilvusInterface:
    def __init__(self, dbname, user='', password=''):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None
        self.connect_server()
        pass

    def connect_server(self):
        # self.conn = psycopg.connect(
        #     f"dbname={self.dbname} user={self.user} password={self.password}"
        # )
        # self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        # register_vector(self.conn)
        conn = connections.connect(host="127.0.0.1", port=19530)
        # database = db.create_database("my_database")

    def disconnect_server(self):
        # self.conn.close()
        pass

    def execute_query(self, query):
        # result = self.conn.execute(query).fetchall()
        # self.conn.commit()
        # return result
        pass

    def create_table(self, table_name, vector_size):
        # query = f'''CREATE TABLE IF NOT EXISTS {table_name}
        #  (id bigserial PRIMARY KEY, embedding vector({vector_size}))'''
        # self.conn.execute(query)
        # self.conn.commit()
        pass

    def drop_table(self, table_name):
        # query = "DROP TABLE IF EXISTS " + table_name
        # self.conn.execute(query)
        # self.conn.commit()
        pass

    def get_size_of_table(self, table_name):
        # query = f"""SELECT
        #  pg_size_pretty( pg_total_relation_size('{table_name}'))"""
        # result = self.conn.execute(query).fetchall()
        # self.conn.commit()
        # print(result)
        # return result[0][0]
        pass

    def insert_single_vector(self, table_name, vector):
        # query = f'INSERT INTO {table_name} (embedding) VALUES (%s)'
        # self.conn.execute(query, (vector,))
        # self.conn.commit()
        pass

    def insert_vector_from_csv(self, table_name, csv_path):
        # df = pd.read_csv(csv_path)
        # df = df.to_numpy()

        # for i in range(df.shape[1]):
        #     vector = df[:, i]
        #     self.insert_single_vector(table_name, vector)
        pass

    def get_rows_cnt(self, table_name):
        # query = f'SELECT COUNT(*) FROM {table_name}'
        # result = self.conn.execute(query).fetchall()
        # print(result)
        # self.conn.commit()
        # return result[0][0]
        pass
