import psycopg

# import numpy as np
import pandas as pd
from pgvector.psycopg import register_vector


class PGvectorInterface:
    def __init__(self, dbname, user, password=''):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None
        self.connect_server()
        pass

    def connect_server(self):
        self.conn = psycopg.connect(
            f"dbname={self.dbname} user={self.user} password={self.password}"
        )
        self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.conn)

    def disconnect_server(self):
        self.conn.close()

    def execute_query(self, query):
        result = self.conn.execute(query).fetchall()
        self.conn.commit()
        return result

    def create_table(self, table_name, dimention,
                     metrix=None, index_types=None):
        query = f'''CREATE TABLE IF NOT EXISTS {table_name}
         (id bigserial PRIMARY KEY, embedding vector({dimention}))'''
        self.conn.execute(query)
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

        if metrix == 'l2':
            index_flag = 1
            metrix_name = "vector_l2_ops"
        elif metrix == 'cosine':
            index_flag = 1
            metrix_name = "vector_cosine_ops"
        else:
            print("No metrix")

        if index_flag == 1:
            index_query = f"""
            CREATE INDEX ON {table_name}
            USING {index_types} (embedding {metrix_name})"""
            self.conn.execute(index_query)
            # print("create index")
        else:
            print("no indexing")
        self.conn.commit()

    def drop_table(self, table_name):
        query = "DROP TABLE IF EXISTS " + table_name
        self.conn.execute(query)
        self.conn.commit()

    def get_size_of_table(self, table_name):
        query = f"""SELECT
         pg_total_relation_size('{table_name}'), pg_table_size('{table_name}'), 
         pg_indexes_size('{table_name}')"""
        result = self.conn.execute(query).fetchall()
        self.conn.commit()
        # print(result)
        return result[0][0]

    def insert_single_vector(self, table_name, vector):
        query = f'INSERT INTO {table_name} (embedding) VALUES (%s)'
        self.conn.execute(query, (vector,))
        self.conn.commit()

    def transfer_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        data = df.to_numpy()
        data = [data[i, :] for i in range(data.shape[0])]
        return data

    def insert_vector_from_csv(self, table_name, data):

        # print(f"{df.shape = }")
        # print(len(data))
        # print(data[0])
        query = f'INSERT INTO {table_name} (embedding) VALUES (%s)'
        for vector in data:
            self.conn.execute(query, (vector,))
        self.conn.commit()

    def indexing_data(self, table_name, metrix, index_types):
        if metrix == 'l2':
            metrix_name = "vector_l2_ops"
        elif metrix == 'cosine':
            metrix_name = "vector_cosine_ops"
        else:
            return
        index_query = f"""
        CREATE INDEX ON {table_name}
        USING {index_types} (embedding {metrix_name})"""
        self.conn.execute(index_query)
        self.conn.commit()

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
        LIMIT 1
        """
        result = self.conn.execute(sim_query, (embedding_vector,)).fetchall()
        # print(result)
        return result
