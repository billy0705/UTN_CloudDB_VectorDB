import time
import pandas as pd
import json

from pgvector_interface import PGvectorInterface


def get_data_info(csv_path):
    df = pd.read_csv(csv_path)
    df = df.to_numpy()
    return df.shape


test_db_interface = [PGvectorInterface]
test_metrix = {PGvectorInterface: ["l2", "cosine"]}
test_index_type = {PGvectorInterface: ["hnsw", "ivfflat"]}
db_name_dict = {PGvectorInterface: "PGvector"}

name = "vector_test"
csv_path = "./data/clustered_vectors.csv"
shape = get_data_info(csv_path)

test_csv_path = "./data/similarity_vectors.csv"
test_vector = pd.read_csv(test_csv_path)
test_vector = test_vector.to_numpy().flatten()
test_round = 10
db_benchmarks = []

for db_interface in test_db_interface:
    print(db_interface)
    if db_interface == PGvectorInterface:
        db = db_interface('postgres', 'billyslim')
    else:
        pass

    create_time = 0
    insert_time = 0
    similarity_time = 0

    db_BM = {
        "Name": db_name_dict[db_interface],
        "Data-info": {
            "#vector": shape[0],
            "dimension": shape[1]
        },
        "Test round": test_round,
        "Methods": {}
    }

    for i in range(test_round):
        for index_type in test_index_type[db_interface]:
            for metrix in test_metrix[db_interface]:
                t_name = f"{index_type}+{metrix}"
                if i == 0:
                    db_BM["Methods"][t_name] = {}
                    db_BM["Methods"][t_name]["create_time"] = 0
                    db_BM["Methods"][t_name]["insert_time"] = 0
                    db_BM["Methods"][t_name]["similarity_time"] = 0
                    db_BM["Methods"][t_name]["size"] = 0
                print("#"*40)
                print(f"{db_name_dict[db_interface]}")
                print(f"{index_type = } and {metrix = }")

                db.drop_table(name)
                # print(index_type, metrix)

                # create table
                start_time = time.time()
                db.create_table(name, 1000, metrix=metrix,
                                index_types=index_type)
                db_BM["Methods"][t_name]["create_time"] += (time.time() -
                                                            start_time)

                # insert data
                start_time = time.time()
                db.insert_vector_from_csv(name, csv_path)
                db_BM["Methods"][t_name]["insert_time"] += (time.time() -
                                                            start_time)

                # size of table
                db_BM["Methods"][t_name]["size"] += db.get_size_of_table(name)

                # similarity_search
                start_time = time.time()
                for _ in range(1):
                    result = db.similarity_search(name, test_vector, metrix)
                db_BM["Methods"][t_name]["similarity_time"] += (time.time() -
                                                                start_time)

                # others ...

                # print result
                # print(f"Creating table time: {create_time}")
                # print(f"Inserting data time: {insert_time}")
                # print(f"Similarity search time (100 times): {search_time}")
                # print(f"Size of table: {size}")

    print(db_BM)
    db_benchmarks.append(db_BM.copy())


with open("./result.json", 'w', encoding='utf-8') as f:
    json.dump(db_benchmarks, f, ensure_ascii=False, indent=4)
