import os
from tqdm import tqdm
from huggingface_hub import HfApi
api = HfApi()

BENCHMARKS = [
    "air-bench",
    # "classic",
    # "cleva",
    # "decodingtrust",
    # "heim",
    # "lite",
    # "mmlu",
    # "instruct",
    # "image2structure",
    # "safety",
    # "thaiexam",
    # "vhelm",
]

for benchmark in BENCHMARKS:
    for sub_folder in ["releases", "runs"]:
        if not os.path.exists(
            os.path.join(benchmark, sub_folder)
        ):
            continue
        
        all_versions = os.listdir(os.path.join(benchmark, sub_folder))
        for version in all_versions:
            print(f"Uploading {benchmark}/{sub_folder}/{version}")
            
            # all_ffs = os.listdir(os.path.join(benchmark, sub_folder, version))
            # for ff in tqdm(all_ffs):
                
            #     if os.path.isdir(
            #         os.path.join(benchmark, sub_folder, version, ff)
            #     ):
            #         api.upload_folder(
            #             folder_path=os.path.join(benchmark, sub_folder, version, ff),
            #             repo_id="stair-lab/reeval_jsons_full",
            #             repo_type="dataset",
            #             path_in_repo=os.path.join(benchmark, sub_folder, version, ff),
            #         )
            #     else:
            #         api.upload_file(
            #             path_or_fileobj=os.path.join(benchmark, sub_folder, version, ff),
            #             repo_id="stair-lab/reeval_jsons_full",
            #             repo_type="dataset",
            #             path_in_repo=os.path.join(benchmark, sub_folder, version, ff),
            #         )
            
            if os.path.isdir(
                os.path.join(benchmark, sub_folder, version)
            ):
                api.upload_folder(
                    folder_path=os.path.join(benchmark, sub_folder, version),
                    repo_id="stair-lab/reeval_jsons_full",
                    repo_type="dataset",
                    path_in_repo=os.path.join(benchmark, sub_folder, version),
                )
            else:
                api.upload_file(
                    path_or_fileobj=os.path.join(benchmark, sub_folder, version),
                    repo_id="stair-lab/reeval_jsons_full",
                    repo_type="dataset",
                    path_in_repo=os.path.join(benchmark, sub_folder, version),
                )