import psycopg2
from psycopg2.extras import execute_values

import numpy as np
import pandas as pd
from pgvector.psycopg2 import register_vector


class PGvectorInterface:
    def __init__(self, dbname, user, password=''):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None
        self.connect_server()
        pass

    def connect_server(self):
        self.conn = psycopg2.connect(
            f"dbname={self.dbname} user={self.user} password={self.password}"
        )
        self.cur = self.conn.cursor()
        self.cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.conn)

    def disconnect_server(self):
        self.conn.close()

    def execute_query(self, query):
        result = self.conn.execute(query).fetchall()
        return result

    def create_table(self, table_name, dimention,
                     metric=None, index_types=None):
        query = f'''CREATE TABLE IF NOT EXISTS {table_name}
         (id bigserial PRIMARY KEY, embedding vector({dimention}))'''
        self.cur.execute(query)
        index_flag = 0
        if index_types == "hnsw":
            index_flag = 1
            pass
        elif index_types == "ivfflat":
            index_flag = 1
            pass
        else:
            index_flag = 0
            print("No index_types")

        if metric == 'l2':
            index_flag = 1
            metric_name = "vector_l2_ops"
        elif metric == 'cosine':
            index_flag = 1
            metric_name = "vector_cosine_ops"
        else:
            print("No metric")

        if index_flag == 1:
            index_query = f"""
            CREATE INDEX ON {table_name}
            USING {index_types} (embedding {metric_name})"""
            self.cur.execute(index_query)
            # print("create index")
        else:
            print("no indexing")

    def drop_table(self, table_name):
        query = "DROP TABLE IF EXISTS " + table_name
        self.cur.execute(query)

    def get_size_of_table(self, table_name):
        query = f"""SELECT
         pg_total_relation_size('{table_name}'), pg_table_size('{table_name}'),
         pg_indexes_size('{table_name}')"""
        self.cur.execute(query)
        result = self.cur.fetchall()
        # print(result)
        return result[0][0]

    def insert_single_vector(self, table_name, vector):
        query = f'INSERT INTO {table_name} (embedding) VALUES (%s)'
        self.cur.execute(query, (vector,))
        self.conn.commit()

    def transfer_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        data = df.to_numpy()
        data = [(i, np.array(data[i, :])) for i in range(data.shape[0])]
        return data

    def insert_vector_from_csv(self, table_name, data):

        query = f'INSERT INTO {table_name} (id, embedding) VALUES %s'
        execute_values(self.cur, query, data)

    def indexing_data(self, table_name, metric, index_types):
        if metric == 'l2':
            metric_name = "vector_l2_ops"
        elif metric == 'cosine':
            metric_name = "vector_cosine_ops"
        else:
            return
        index_query = f"""
        CREATE INDEX ON {table_name}
        USING {index_types} (embedding {metric_name})"""
        self.cur.execute(index_query)

    def get_rows_cnt(self, table_name):
        query = f'SELECT COUNT(*) FROM {table_name}'
        self.cur.execute(query)
        result = self.cur.fetchall()
        return result[0][0]

    def similarity_search(self, table_name, embedding_vector, metric):
        if metric == "l2":
            symbol = "<->"
        elif metric == "cosine":
            symbol = "<=>"
        else:
            print("Error with metric type")
            return
        sim_query = f"""
        SELECT id, embedding {symbol} (%s) AS distance
        FROM {table_name}
        ORDER BY distance ASC
        LIMIT 3
        """
        self.cur.execute(sim_query, (embedding_vector,))
        result = self.cur.fetchall()
        # print(result)
        return result[0][0], result[0][1]
