from huggingface_hub import HfApi, snapshot_download
import pandas as pd
import torch
import numpy as np
import pickle
from torchmetrics.functional import spearman_corrcoef

dataset = "mmlu"
data_folder = snapshot_download(
    repo_id="stair-lab/reeval_csv", repo_type="dataset"
)
df_responses = pickle.load(open(f"{data_folder}/{dataset}/responses.pkl", "rb"))
instances = pd.read_csv(f"{data_folder}/{dataset}/instances.csv")

# merge df_responses and instances by instance_id
# instance_id in df_responses is float, while in instances is int
# but they are the same
df_responses["instance_id"] = df_responses["instance_id"].astype(int)
df = pd.merge(df_responses, instances, on="instance_id")

# get a list of unique subject
subjects = df["subject"].unique()

# filter out the row of subject[0] and subject[1]
np.random.seed(1)
np.random.shuffle(subjects)
subjects_subset = subjects[0:1]

df_subject = df[df["subject"].isin(subjects_subset)]

# only keep the model_id, instance_id, and exact_match columns
df_subject = df_subject[["model_id", "instance_id", "exact_match"]]

# create a pivot table with model_id as index, instance_id as columns, and exact_match as values
df_subject = df_subject.pivot(index="model_id", columns="instance_id", values="exact_match")

# drop the columns with only 0 or 1 value
df_subject = df_subject.loc[:, (df_subject != 0).any()]
df_subject = df_subject.loc[:, (df_subject != 1).any()]

# check if there is any nan value
print(df_subject.isnull().values.any())

# if there is a nan, fill it with -1
df_subject = df_subject.fillna(-1)

# convert type to int
df_subject = df_subject.astype(int)

# save the data to csv file without index and header and type int
df_subject.to_csv(f"Subset.csv", index=False, header=False)

# save the file to .pt format and pust to huggingface hub to check with 
# python calibration
df_subject = torch.tensor(df_subject.values)
torch.save(df_subject, f"response_matrix.pt")

api = HfApi()
api.upload_file(
    repo_id="stair-lab/reeval_matrices",
    path_in_repo="mmlu0/response_matrix.pt",
    path_or_fileobj="response_matrix.pt",
    repo_type="dataset"
)

# read the file again to check if it is saved correctly
data_folder = snapshot_download(
    repo_id="stair-lab/reeval_matrices", repo_type="dataset"
)
response_matrix = torch.load(f"{data_folder}/mmlu0/response_matrix.pt")
######################################################################

# Run the mirt.R script to get the item_difficulty and ability_MAP
import subprocess
subprocess.run(["Rscript", "mirt.R"])

# load item_difficulty and ability_MAP csv into a torch tensor
item_difficulty = pd.read_csv(f"item_difficulty.csv", header=None)
ability = pd.read_csv(f"ability_MAP.csv", header=None)
item_difficulty = torch.tensor(item_difficulty.values)
item_difficulty = item_difficulty[:, 1]
ability = torch.tensor(ability.values)[:, 0]

item_difficulty_python = pd.read_csv(f"difficulties_python.csv", header=None)
ability_python = pd.read_csv(f"abilities_python.csv", header=None)

item_difficulty_python = torch.tensor(item_difficulty_python.values)
ability_python = torch.tensor(ability_python.values)

# mean score across users 
item_difficulty_ctt = df_subject.float().mean(axis=0)
torch.corrcoef(torch.stack([item_difficulty, item_difficulty_ctt]))
torch.corrcoef(torch.stack([item_difficulty_python.squeeze(), item_difficulty_ctt]))

# spearman correlation
spearman_corrcoef(item_difficulty, item_difficulty_ctt)

diff = torch.stack([item_difficulty.flatten(), item_difficulty_python.flatten()])
corr = torch.corrcoef(diff)
print(corr)

diff = diff.numpy()
import matplotlib.pyplot as plt
plt.scatter(diff[0], diff[1])
plt.savefig("difficulties.png", dpi=300, bbox_inches="tight")

breakpoint()



print("Correlation between item_difficulty:", corr[0, 1])
corr = np.corrcoef(ability.values.flatten(), ability_python.flatten())
print("Correlation between ability:", corr[0, 1])

n_item = item_difficulty.shape[0]
n_testtaker = ability.shape[0]

item_difficulty = item_difficulty[None, :].repeat(ability.shape[0], 1)
ability = ability[:, None].repeat(1, item_difficulty.shape[1])
prob_correct = torch.sigmoid(ability - item_difficulty).numpy()

# compute AUC ROC with prob = prob_correct and ground truth = df_subject
from sklearn.metrics import roc_auc_score
y_true = df_subject.values.flatten()
y_score = prob_correct.flatten()
print("AUC ROC:", roc_auc_score(y_true, y_score))
print("Brier score:", np.mean((y_true - y_score) ** 2))