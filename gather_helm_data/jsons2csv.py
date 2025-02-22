import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join, exists

def infer_column_types(df):
    inferred_types = {}
    for col in df.columns:
        try:
            unique_values = df[col].dropna().unique()
        except:
            df[col] = df[col].apply(lambda x: json.dumps(x))
            unique_values = df[col].dropna().unique()
        
        # Check if all values are boolean-like
        if set(unique_values).issubset({"True", "False", "0", "1"}):
            inferred_types[col] = "bool"
            df[col] = df[col].map(lambda x: True if x in ["True", "1"] else False).astype("bool")

        # Check if all values can be converted to integers (efficiently)
        elif np.all(~pd.isna(pd.to_numeric(unique_values, errors="coerce"))):
            inferred_types[col] = "int"
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

        # Check if it's categorical (based on fraction of unique values)
        elif df[col].nunique() / len(df) < 0.1:
            inferred_types[col] = "categorical"
            df[col] = df[col].astype("string").astype("category")

        else:
            inferred_types[col] = "string"

    return inferred_types

if __name__ == "__main__":
    os.makedirs(f"CSV", exist_ok=True)
    BENCHMARKS = ["air-bench", "classic", "thaiexam", "mmlu"] # ["classic"]
    lo = lambda x: json.load(open(x, "r"))

    all_paths = []
    for benchmark in BENCHMARKS:
        dir_path = f"helm_jsons/{benchmark}/releases"
        assert exists(dir_path)
        latest_release = sorted(os.listdir(dir_path))[-1]
        folder_dict = lo(f"{dir_path}/{latest_release}/runs_to_run_suites.json")
        paths = [join("helm_jsons", benchmark, "runs", suite, run) for run, suite in folder_dict.items()]
        all_paths += paths

    files = ["display_requests.json", "display_predictions.json", "run_spec.json"]
    all_paths = [p for p in tqdm(all_paths) if all([exists(f"{p}/{f}") for f in files])]
    all_lists = [[lo(f"{p}/{f}") for p in tqdm(all_paths)] for f in files]
    # list of all files within the benchmark, organzed by the 3 group of files

    results = []
    for d_requests, d_predictions, run_specs, paths in tqdm(zip(*all_lists, all_paths), total=len(all_lists[0])):
        d_requests = pd.json_normalize(d_requests)
        d_predictions = pd.json_normalize(d_predictions)
        run_specs = pd.json_normalize(run_specs)
        run_specs["benchmark"] = paths.split("/")[1]
        run_specs = run_specs.loc[run_specs.index.repeat(d_predictions.shape[0])].reset_index(drop=True)
        overlap_column = d_predictions.columns.intersection(d_requests.columns)
        d_requests = d_requests.drop(columns=overlap_column)
        display_both = pd.concat([d_requests, d_predictions, run_specs], axis=1)
        results.append(display_both)

    results = pd.concat(results, axis=0, join='outer')
    infer_column_types(results)
    results.reset_index(drop=True, inplace=True)

    for col in results.columns:
        if results[col].dtype != "category" and np.isnan(results[col]).all():
            results = results.drop(columns=col)
        else:
            if results[col].dtype == "float64" and np.nanmax(results[col]) < 65500 and np.nanmin(results[col]) > -65500:
                results[col] = results[col].astype("float16")

    results.info()
    results.to_pickle("CSV/responses.pkl")
    df = pd.read_pickle("CSV/responses.pkl")
