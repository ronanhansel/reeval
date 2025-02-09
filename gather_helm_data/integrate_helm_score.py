import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
from configs import WINRATE4GROUP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    args = parser.parse_args()
    
    group_info = json.load(open(os.path.join("CSV", args.benchmark, "group_infos.json")))
    
    for group, datasets in group_info.items():
        winrate_key = WINRATE4GROUP[group]
        winrate_df = pd.read_csv(
            os.path.join(
                "CSV", args.benchmark, "groups", f"{group}_winrate.csv"
            )
        )
        
        # Create a mapping between model_name and winrate
        model_name2winrate = {}
        for _, row in winrate_df.iterrows():
            model_name = row["model_name"]
            winrate = row[winrate_key]
            model_name2winrate[model_name] = winrate
        
        for dataset in datasets:
            model_key_df = pd.read_csv(
                os.path.join(
                    "matrices", args.benchmark, dataset, "model_keys.csv"
                )
            )
            list_winrate = []
            for _, row in model_key_df.iterrows():
                model_name = row["display_name"]
                if model_name in model_name2winrate:
                    list_winrate.append(model_name2winrate[model_name])
                else:
                    list_winrate.append("")
                    
            if "winrate" in model_key_df.columns:
                del model_key_df["winrate"]
            model_key_df["helm_score"] = list_winrate
            model_key_df.to_csv(
                os.path.join(
                    "matrices", args.benchmark, dataset, "model_keys.csv"
                ), index=False
            )
            