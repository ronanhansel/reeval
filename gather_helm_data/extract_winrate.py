import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
from functools import reduce
from jsons2csv import get_run_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    args = parser.parse_args()
    
    benchmark = args.benchmark
    if os.path.exists(
        os.path.join(benchmark, "releases")
    ):
        latest_release = sorted(
            os.listdir(os.path.join(benchmark, "releases"))
        )[-1]
        run_path = os.path.join(
            benchmark, "releases", latest_release
        )

    else:
        latest_release = sorted(
            os.listdir(os.path.join(benchmark, "runs"))
        )[-1]
        run_path = os.path.join(
            benchmark, "runs", latest_release
        )
        
    if not os.path.exists(run_path):
        print("No groups found in the benchmark")
        sys.exit(1)
        
    
    all_groups_results = os.listdir(
        os.path.join(run_path, "groups")
    )
    all_groups_results = [x for x in all_groups_results if x.endswith(".json")]
    groups_has_wr = []
    
    for group_json in all_groups_results:
        group_results = json.load(
            open(os.path.join(run_path, "groups", group_json))
        )
        
        group_name = group_json.split(".")[0]
        
        list_winrate_df = []
        for aspect in group_results:
            aspect_name = aspect["title"].lower().replace(" ", "_")
            if aspect["header"][1]["value"] != "Mean win rate":
                print(f"Group {group_name} - {aspect_name} does not have win rate!")
                continue
            
            winrate_dict = {
                "model_name": [],
                f"winrate_{aspect_name}": []
            }
            for model_info in aspect["rows"]:
                model_name = model_info[0]["value"]
                if "value" not in model_info[1]:
                    continue
                winrate = model_info[1]["value"]
                
                winrate_dict["model_name"].append(model_name)
                winrate_dict[f"winrate_{aspect_name}"].append(winrate)
                
            winrate_df = pd.DataFrame(winrate_dict)
            list_winrate_df.append(winrate_df)
            
        if len(list_winrate_df) == 0:
            continue
        
        winrate_df = reduce(
            lambda x, y: pd.merge(x, y, on="model_name", how="outer"),
            list_winrate_df
        )
        groups_has_wr.append(group_name)
        
        # Save winrate_df to csv
        os.makedirs(f"CSV/{benchmark}/groups", exist_ok=True)
        winrate_df.to_csv(
            f"CSV/{benchmark}/groups/{group_name}_winrate.csv",
            index=False
        )
        
        print(f"Group {group_name} winrate saved to CSV")
        
    
    # Get list of datasets for each group
    group_dataset_dict = {}
    all_run_paths = get_run_path(benchmark)
    for run_path in tqdm(all_run_paths):
        scenario_info = json.load(
            open(os.path.join(run_path, "scenario.json"))
        )
        final_folder = run_path.split("/")[-1]
        if not final_folder.startswith("legal_support"):
            name = final_folder.split(":")[0]
        else:
            name = final_folder.split(",")[0]
        groups = scenario_info["tags"]
        for group in groups:
            if group not in groups_has_wr:
                continue
            if group not in group_dataset_dict:
                group_dataset_dict[group] = []
            if name not in group_dataset_dict[group]:
                group_dataset_dict[group].append(name)
            
    # Save group_dataset_dict to json
    with open(f"CSV/{benchmark}/group_infos.json", "w") as f:
        json.dump(group_dataset_dict, f, indent=4)
    