import os
from data_generation import generate_dataset
from benchmark import Benchmark


# Function to check and create directory
def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


# Function to format dataset names
def format_dataset_name(name):
    return name.replace(" ", "_").lower()


# Main function
def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_dir, "data")

    # Step 1: Check if 'data' directory exists, create if not
    check_and_create_dir(data_dir)

    # Dataset configurations
    datasets = {
        "Small Dataset": {"num_vectors": 1000, "num_dim": 1000},
        "Large Dataset": {"num_vectors": 10000, "num_dim": 1000},
    }

    for dataset_name, params in datasets.items():
        formatted_name = format_dataset_name(dataset_name)
        dataset_path = os.path.join(data_dir, formatted_name)

        # Step 2: Create dataset directory
        if not check_and_create_dir(dataset_path):
            continue

        # Step 3: Generate dataset
        generate_dataset(params["num_vectors"], params["num_dim"],
                         dataset_path)
        print(f"Generated dataset {dataset_name} saved to {dataset_path}")
        # Paths for data and test CSV files
        data_csv_path = os.path.join(dataset_path, "data.csv")
        test_csv_path = os.path.join(dataset_path, "test.csv")

        # Step 4: Run benchmark test
        result_file = os.path.join(repo_dir, "result",
                                   f"{formatted_name}_result.json")
        check_and_create_dir(os.path.dirname(result_file))
        try:
            Benchmark(data_csv_path, test_csv_path, result_file=result_file,
                      pg_dbname="", pg_username="")
            print(
                f"Benchmark results for {dataset_name} saved to {result_file}")
        except Exception as e:
            print("Benchmark was not able to finish calculating " +
                  f"and generating results with error {e}")
            print("Try to run the Benchmark from GUI and change the settings.")
            continue


if __name__ == "__main__":
    main()
