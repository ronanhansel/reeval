import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder",
    type=str,
    default="data", # result
)
args = parser.parse_args()

folder = args.folder
repo_id = f"stair-lab/reeval_{folder}_public"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=folder,
)

print(f"Downloaded `{repo_id}` into `./{folder}`")
