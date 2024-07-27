import time
import pandas as pd
import json

from pgvector_interface import PGvectorInterface
from milvus_interface import MilvusInterface


def get_data_info(csv_path):
    df = pd.read_csv(csv_path)
    df = df.to_numpy()
    return df.shape


test_db_interface = [MilvusInterface, PGvectorInterface]
test_metrix = {
    PGvectorInterface: ["l2", "cosine"],
    MilvusInterface: ["L2", "COSINE"]
}
test_index_type = {
    PGvectorInterface: ["hnsw", "ivfflat"],
    MilvusInterface: ["HNSW", "FLAT"]
}
db_name_dict = {
    PGvectorInterface: "PGvector",
    MilvusInterface: "Milvus"
}


def benchmark_test(i, index_type, metrix, db_BM,
                   db, collection_name, csv_path, test_vector):
    t_name = f"{index_type}+{metrix}"
    if i == 0:
        db_BM["Methods"][t_name] = {}
        db_BM["Methods"][t_name]["create_time"] = 0
        db_BM["Methods"][t_name]["insert_time"] = 0
        db_BM["Methods"][t_name]["similarity_time"] = 0
        db_BM["Methods"][t_name]["size"] = 0
    print(f"Round {i+1} start")

    db.drop_table(collection_name)
    # print(index_type, metrix)

    # create table
    start_time = time.time()
    db.create_table(collection_name, 1000, metrix=metrix,
                    index_types=index_type)
    db_BM["Methods"][t_name]["create_time"] += (time.time() -
                                                start_time)

    # prepare data
    data = db.transfer_csv(csv_path)
    # insert data
    start_time = time.time()
    db.insert_vector_from_csv(collection_name, data)
    db_BM["Methods"][t_name]["insert_time"] += (time.time() -
                                                start_time)

    # size of table
    db_BM["Methods"][t_name]["size"] += db.get_size_of_table(
        collection_name
    )

    # similarity_search
    start_time = time.time()
    for test_i in range(test_vector.shape[0]):
        _ = db.similarity_search(
            collection_name,
            test_vector[test_i, :],
            metrix
        )
    db_BM["Methods"][t_name]["similarity_time"] += (
        time.time() - start_time
    )

    # others ...

    # print results

    return db_BM


def Benchmark(
    csv_path,
    test_csv_path,
    test_round=1,
    collection_name="vector_benchmark_test",
    result_file="./result/result.json",
    pg_dbname='postgres',
    pg_username='billyslim',
    pg_password='',
    milvus_db_path='milvus_db/milvus_demo.db'
):
    train_data_shape = get_data_info(csv_path)
    test_data_shape = get_data_info(test_csv_path)
    test_vector = pd.read_csv(test_csv_path)
    test_vector = test_vector.to_numpy()
    db_benchmarks = []

    print("Start Benchmark process")
    print("#"*40)
    print("Informaton of dataset")

    print(f"""Training dataset:
    # vector = {train_data_shape[0]}
    dimension = {train_data_shape[1]}""")
    print(f"""Testing dataset:
    # vector = {test_data_shape[0]}
    dimension = {test_data_shape[1]}""")

    total_start_time = time.time()

    for db_interface in test_db_interface:
        # print(db_name_dict[db_interface])
        if db_interface == PGvectorInterface:
            db = db_interface(pg_dbname, pg_username)
        elif db_interface == MilvusInterface:
            db = db_interface(milvus_db_path)
        else:
            continue

        db_BM = {
            "Name": db_name_dict[db_interface],
            "Train-Data-info": {
                "#vector": train_data_shape[0],
                "dimension": train_data_shape[1]
            },
            "Test-Data-info": {
                "#vector": test_data_shape[0],
                "dimension": test_data_shape[1]
            },
            "Test round": test_round,
            "Methods": {}
        }

        for index_type in test_index_type[db_interface]:
            for metrix in test_metrix[db_interface]:
                for i in range(test_round):
                    round_strat_time = time.time()
                    print("#"*40)
                    print(f"{db_name_dict[db_interface]}")
                    print(f"{index_type = } and {metrix = }")
                    db_BM = benchmark_test(i, index_type, metrix,
                                           db_BM,
                                           db, collection_name,
                                           csv_path, test_vector)
                    print(f"Round {i+1} spent {time.time()-round_strat_time}")

        db.drop_table(collection_name)
        db.disconnect_server()
        # print(db_BM)
        db_benchmarks.append(db_BM.copy())

    print("#"*40)
    print(f"Total process time: {time.time() - total_start_time}")

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(db_benchmarks, f, ensure_ascii=False, indent=4)

    return 0


if __name__ == "__main__":
    csv_path = "./data/clustered_vectors_small.csv"
    test_csv_path = "./data/clustered_vectors_test.csv"
    result_file = "./result/result_small.json"
    Benchmark(csv_path, test_csv_path, result_file=result_file)
