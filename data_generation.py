# Create it for the dataset benchmark

import numpy as np 
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def generate_dataset(num_vectors, num_dimensions, folder_path,
                     cluster=True, parquet=False):
    # if random (clustered not checked)
    num_test = int(num_vectors * 0.01)
    vectors = np.random.rand(num_vectors, num_dimensions)
    test_vectors = np.random.rand(num_test, num_dimensions)
    if cluster:
        num_clusters = 3
        cluster_cen = np.random.rand(num_clusters, num_dimensions)
        # assign each vector to a cluster and create an arraty for the clustered vectors
        clustered_vectors = np.zeros((num_vectors, num_dimensions))
        test_clustered_vectors = np.zeros((num_test, num_dimensions))
        # assign vectors to clusters
        last = 0
        last_test = 0
        for i in range(num_clusters):
            sigma = np.random.rand(num_dimensions)*0.5
            num_points = int(num_vectors*0.9/num_clusters)
            num_test_points = int(num_test*0.9/num_clusters)
            clustered_vectors[last:num_points + last, :] = np.random.normal(cluster_cen[i], sigma, size=(num_points, num_dimensions))
            test_clustered_vectors[last_test:num_test_points + last_test, :] = np.random.normal(cluster_cen[i], sigma, size=(num_test_points, num_dimensions))
            last += num_points
            last_test += num_test_points
        clustered_vectors[last:last + num_vectors, :] = np.random.rand(num_vectors-last, num_dimensions)
        test_clustered_vectors[last_test:last_test + num_test, :] = np.random.rand(num_test-last_test, num_dimensions)
        # Shuffle the clustered vectors
        np.random.shuffle(clustered_vectors)
        np.random.shuffle(test_clustered_vectors)
        vectors = clustered_vectors
        test_vectors = test_clustered_vectors

    our_df = pd.DataFrame(vectors)
    our_test_df = pd.DataFrame(test_vectors)
    our_df.to_csv(f'{folder_path}/data.csv', index=False, header=True)
    our_test_df.to_csv(f'{folder_path}/test.csv', index=False, header=True)
    if parquet:
        # Create the train and test DataFrames
        train_vectors = vectors
        test_vectors = test_vectors

        # Create the train DataFrame
        train_df = pd.DataFrame({
            'id': np.arange(1, len(train_vectors) + 1),
            'emb': train_vectors.tolist()
        })

        # Create the test DataFrame
        test_df = pd.DataFrame({
            'id': np.arange(1, len(test_vectors) + 1),
            'emb': test_vectors.tolist()
        })
        # Save the DataFrames to Parquet files
        train_table = pa.Table.from_pandas(train_df)
        test_table = pa.Table.from_pandas(test_df)
        pq.write_table(train_table, f'{folder_path}/train.parquet')
        pq.write_table(test_table, f'{folder_path}/test.parquet')
    return
