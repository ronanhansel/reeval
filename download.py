from huggingface_hub import snapshot_download

repo_id = f"stair-lab/reeval"
folder = "data"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=folder,
)

print(f"Downloaded `{repo_id}` into `./{folder}`")
