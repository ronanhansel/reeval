import os
import torch
from configs import TASK2METRICS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = "matrices"
total_datapoints = 0

for benchmark in TASK2METRICS:
    for dataset in TASK2METRICS[benchmark]:
        response_matrix = torch.load(f"{data_folder}/{benchmark}/{dataset}/response_matrix.pt").to(
            device=device, dtype=torch.float32
        )
        total_questions = response_matrix.shape[1]

        print(f"Dataset: {dataset}, Total questions: {total_questions}")
        total_datapoints += total_questions
            
print(f"Total datapoints: {total_datapoints}")
        
