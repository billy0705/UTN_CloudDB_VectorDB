import json
import matplotlib.pyplot as plt
import numpy as np


# read JSON
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# extract data
def extract_data(data, metric):
    results = {}
    methods = set()
    for item in data:
        database = item['Name']
        for method, method_results in item['Methods'].items():
            if database not in results:
                results[database] = {}
            results[database][method] = method_results[metric]
            methods.add(method)
    return results, methods


def generate_figure(data, methods, title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 8))

    databases = list(data.keys())
    x = np.arange(len(databases))  # to use as plot location
    bar_width = 0.1

    for i, method in enumerate(sorted(methods)):
        values = [data[database].get(method, 0) for database in databases]
        if any(values):
            ax.bar(x + i * bar_width, values, bar_width, label=method)

    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(databases)
    ax.set_xlabel('Vector Database Frameworks')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return fig


metrics_labels = {
    'create_time': ('Create Time Comparison', 'Time (s)'),
    'insert_time': ('Insert Time Comparison', 'Time (s)'),
    'similarity_time': ('Similarity Time Comparison', 'Time (s)'),
    'size': ('Size Comparison', 'Size (bytes)'),
    'total_distance': ('Distance (Error)', 'Search Time (s)')
}


def generate_figure_quality(data, data2, methods, title, ylabel, metric="L2"):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    ax_l2, ax_cosine = axs

    for db in data.keys():
        for method in data[db].keys():
            if method.split("+")[1] == "L2":
                ax_l2.scatter(data[db][method], data2[db][method], label=f"{db}+{method}")
            elif method.split("+")[1] == "COSINE":
                ax_cosine.scatter(data[db][method], data2[db][method], label=f"{db}+{method}")

    ax_l2.set_xlabel('Total Distance')
    ax_l2.set_ylabel(ylabel)
    ax_l2.set_title(f"{title} - L2 Metric")
    ax_l2.legend()
    ax_l2.grid(True)

    ax_cosine.set_xlabel('Total Distance')
    ax_cosine.set_title(f"{title} - COSINE Metric")
    ax_cosine.legend()
    ax_cosine.grid(True)

    return fig


def get_plot_figure(metric, file_path):
    data = read_json(file_path)
    title, ylabel = metrics_labels[metric]
    data_extracted, methods = extract_data(data, metric)
    if metric != 'total_distance':
        fig = generate_figure(data_extracted, methods, title, ylabel)
    else:
        data_extracted2, methods = extract_data(data, "similarity_time")
        fig = generate_figure_quality(data_extracted, data_extracted2,
                                      methods, title, ylabel)
    return fig
