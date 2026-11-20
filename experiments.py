import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from clustpy.metrics.external.external_clustering_metrics import (
    fair_normalized_mutual_information,
    unsupervised_clustering_accuracy,
)
from sklearn.metrics.cluster import adjusted_rand_score

from data import (
    get_bbcnews,
    get_bbcsports,
    get_breast_cancer,
    get_dermatology,
    get_ecoli,
    get_glass,
    get_iris,
    get_mnist,
    get_moons,
    get_optdigits,
    get_parkinsons,
    get_reuters,
    get_segmentation,
    get_usps,
    get_waveform,
    get_wine,
)
from models.acc import ACC
from models.bacon import BACON
from models.cobras import COBRAS
from models.ncobras import NCOBRAS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Results output path.",
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.05,
        help="Noise level (probability of oracle mistake).",
    )

    parser.add_argument(
        "-b",
        "--budget",
        type=int,
        default=10,
        help="Query budget.",
    )

    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=int,
        default=None,
        help="Dataset selection.",
    )

    parser.add_argument(
            "--experiment",
            type=str,
            default="compare",
            help="Select experiment to run.",
        )

    args = parser.parse_args()
    return args


class ClusterMethod:
    def __init__(self, name: str, method, params: dict | None = None):
        self.name = name
        self.method = method
        if params is None:
            params = {}
        self.params = params


def get_random_seeds(random_state, n_repeats):
    return random_state, random_state.randint(0, 999999, size=n_repeats)


def compute_metrics(y, preds):
    ari = adjusted_rand_score(y, preds)
    acc = unsupervised_clustering_accuracy(y, preds)
    fnmi = fair_normalized_mutual_information(y, preds)
    return ari, acc, fnmi


def write_row_to_csv(results, save_path):
    df = pd.DataFrame([results])  # dataframe with one row
    # append results to a file, creates a new file if one does not exists
    df.to_csv(
        save_path,
        mode="a",
        header=not os.path.exists(save_path),
        index=False,
    )


def write_row_in_csv(dataset_name, model_name, n_queries, job_index, ari, acc, fnmi, runtime, noise_level, params, csv_path):
    res = {"dataset": dataset_name, "model": model_name, "queries": n_queries, "run-index": job_index, "ari": ari, "accuracy": acc, "fnmi": fnmi, "runtime_total": runtime, "noise_level": noise_level, "params": params}
    write_row_to_csv(res, csv_path)


def eval_ncobras(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path):
    parameters = method.params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
    parameters["random_state"] = random_seed

    time_start = time.time()
    model = method.method(**parameters)
    model.fit(X, y)
    runtime = time.time() - time_start

    # add baseline of zero queries (all points in the same cluster)
    labels = np.zeros_like(y, dtype=int)
    ari, acc, fnmi = compute_metrics(y, labels)
    write_row_in_csv(dataset_name, method.name, 0, repeat_index, ari, acc, fnmi, runtime, noise_level, params_str, csv_path)

    for labels, q in zip(model.labels_hist, model.queries_hist):
        ari, acc, fnmi = compute_metrics(y, np.array(labels))
        write_row_in_csv(dataset_name, method.name, q, repeat_index, ari, acc, fnmi, runtime, noise_level, params_str, csv_path)


def eval_cobras(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path):
    parameters = method.params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
    # parameters["random_state"] = random_seed

    time_start = time.time()
    model = method.method(**parameters)
    model.fit(X, y)
    runtime = time.time() - time_start

    # add baseline of zero queries (all points in the same cluster)
    labels = np.zeros_like(y, dtype=int)
    ari, acc, fnmi = compute_metrics(y, labels)
    write_row_in_csv(dataset_name, method.name, 0, repeat_index, ari, acc, fnmi, runtime, noise_level, params_str, csv_path)

    queries = range(1, len(model.intermediate_clusterings)+1)

    for q, labels in zip(queries, model.intermediate_clusterings):
        ari, acc, fnmi = compute_metrics(y, np.array(labels))
        write_row_in_csv(dataset_name, method.name, q, repeat_index, ari, acc, fnmi, runtime, noise_level, params_str, csv_path)


def eval_acc(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path):
    parameters = method.params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
    parameters["random_state"] = random_seed

    time_start = time.time()
    model = method.method(**parameters)
    model.fit(X, y)
    runtime = time.time() - time_start

    for q, labels in enumerate(model.history):
        ari, acc, fnmi = compute_metrics(y, np.array(labels))
        write_row_in_csv(dataset_name, method.name, q, repeat_index, ari, acc, fnmi, runtime, noise_level, params_str, csv_path)


def eval_bacon(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path):
    parameters = method.params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
    parameters["random_state"] = random_seed

    time_start = time.time()
    model = method.method(**parameters)
    model.fit(X, y)
    runtime = time.time() - time_start

    label_hist = model.labels_history
    for q, labels_macro in enumerate(label_hist):
        if parameters["mode"] == "kmeans":
            # full labeling from super-instances to BACON labels
            labels = labels_macro[model.super_instance_labeling]
        else:
            labels = labels_macro
        ari, acc, fnmi = compute_metrics(y, labels)
        write_row_in_csv(dataset_name, method.name, q, repeat_index, ari, acc, fnmi, runtime, noise_level, params_str, csv_path)


def compare_algs(methods, datasets, repeats, output_path, noise_level, random_state):
    for dataset in datasets:
        dataset_name = dataset[0]
        X, y = dataset[1]()

        _, random_seeds = get_random_seeds(random_state, repeats)
        for repeat_index in range(repeats):
            print(f"Evaluating: {dataset_name} [{repeat_index+1}/{repeats}]")
            csv_path = output_path / f"{dataset_name}.csv"
            random_seed = random_seeds[repeat_index]
            time_start = time.time()
            for method in methods:
                try:
                    if method.name == "NCOBRAS":
                        eval_ncobras(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path)
                    elif method.name == "COBRAS":
                        eval_cobras(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path)
                    elif "ACC" in method.name:
                        eval_acc(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path)
                    elif "BACON" in method.name:
                        eval_bacon(method, X, y, dataset_name, repeat_index, noise_level, random_seed, csv_path)
                    else:
                        raise(ValueError(f"Unknown method: {method.name}!"))
                except Exception:
                    print("Ran into an error while evaluating the method")
                    traceback.print_exc()
                    
            runtime_rep = time.time() - time_start
            print(f"Repeat finished in {runtime_rep:.1f} seconds")

if __name__ == "__main__":

    args = parse_args()

    n_repeats = args.repeats
    noise_level = args.epsilon
    query_budget = args.budget

    random_state = np.random.RandomState(42)

    # create results directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_loaders = {
        "moons": ("Moons", get_moons),
        "iris": ("Iris", get_iris),
        "wine": ("Wine", get_wine),
        "breastcancer": ("breastCancer", get_breast_cancer),
        "derma": ("Dermatology", get_dermatology),
        "glass": ("Glass", get_glass),
        "parkinsons": ("Parkinsons", get_parkinsons),
        "ecoli": ("Ecoli", get_ecoli),
        "segment": ("Segmentation", get_segmentation),
        "waveform": ("Waveform", get_waveform),
        "optdigits": ("Optdigits", get_optdigits),
        "usps": ("USPS", get_usps),
        "mnist": ("MNIST", get_mnist),
        "reuters": ("Reuters", get_reuters),
        "bbcnews": ("BBCNews", get_bbcnews),
        "bbcsports": ("BBCSports", get_bbcsports),
    }

    selected_data = {
        0: "moons",
        1: "iris",
        2: "wine",
        3: "breastcancer",
        4: "derma",
        5: "glass",
        6: "parkinsons",
        7: "ecoli",
        8: "segment",
        9: "waveform",
        10: "optdigits",
        11: "usps",
        12: "mnist",
        13: "reuters",
        14: "bbcnews",
        15: "bbcsports",
    }

    if args.dataset is not None:
        dataset_sel = selected_data[args.dataset]
        print("Selecting dataset", dataset_sel)
        datasets = [data_loaders[dataset_sel]]
    else:
        datasets = [item for item in data_loaders.values()]

    if args.experiment == "compare":
        methods = [
            ClusterMethod("BACON", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6, "sample_eps": True}),
            ClusterMethod("BACON-random", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6, "sample_eps": True, "random_queries": True}),
            ClusterMethod("COBRAS", COBRAS, {"epsilon": noise_level, "max_queries": query_budget}),
            ClusterMethod("NCOBRAS", NCOBRAS, {"epsilon": noise_level, "max_queries": query_budget}),
            ClusterMethod("ACC", ACC, {"noise_level": noise_level, "max_queries": query_budget, "aquisition_function": "maxexp", "init_type":"kmeans", "n_super_instances": 50}),
        ]

    elif args.experiment == "compression":
        methods = [
            ClusterMethod("BACON-full", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "full", "knn": 11, "sample_eps": True, "exact_co_clustering": True}),
            ClusterMethod("BACON-micro20", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 20, "knn": 6, "sample_eps": True}),
            ClusterMethod("BACON-micro30", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 30, "knn": 6, "sample_eps": True}),
            ClusterMethod("BACON-micro40", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 40, "knn": 6, "sample_eps": True}),
            ClusterMethod("BACON-micro50", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6, "sample_eps": True}),
        ]
    elif args.experiment == "knn":
        methods = [
            ClusterMethod("BACON-knn2", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 3, "sample_eps": True}),
            ClusterMethod("BACON-knn5", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6, "sample_eps": True}),
            ClusterMethod("BACON-knn10", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 11, "sample_eps": True}),
            ClusterMethod("BACON-knn20", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 21, "sample_eps": True}),
        ]

    elif args.experiment == "noise":
        methods = [
            ClusterMethod("BACON", BACON, {"epsilon": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6, "sample_eps": True}),
            ClusterMethod("BACON-fixed", BACON, {"epsilon": noise_level, "epsilon_init": noise_level, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6}),
            ClusterMethod("BACON-fixed-low", BACON, {"epsilon": noise_level, "epsilon_init": 0.01, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6}),
            ClusterMethod("BACON-fixed-high", BACON, {"epsilon": noise_level, "epsilon_init": 0.3, "budget": query_budget, "mode": "kmeans", "n_super_instances": 50, "knn": 6}),
        ]

    compare_algs(methods, datasets, n_repeats, output_dir, noise_level, random_state)
