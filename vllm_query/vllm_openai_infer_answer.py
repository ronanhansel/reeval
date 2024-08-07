import os
from huggingface_hub import login
import datasets
from bacher_modify import GPTBatcher
from dotenv import load_dotenv
import csv
import argparse

parser = argparse.ArgumentParser(description='Generate answers and save to CSV.')
parser.add_argument('--url', type=str, required=True, help='API base URL')
parser.add_argument('--start', type=int, required=True, help='Start index')
parser.add_argument('--end', type=int, required=True, help='End index')

args = parser.parse_args()
url = args.url
start = args.start
end = args.end

login(token = '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')
airbench = datasets.load_dataset("stanford-crfm/air-bench-2024", split="test")

model_string_list = ['meta-llama/Meta-Llama-3-8B-Instruct',]
subset_model_string_list = model_string_list[start-1:end]

for model_string in subset_model_string_list:
    batcher_answer = GPTBatcher(api_key="EMPTY",
                            model_name=model_string,
                            temperature=0,
                            num_workers=64,
                            max_token=128,
                            api_base_url=url,
                            )

    row_list = []
    question_list = []
    for i in range(3000):
        row_list.append([airbench[i]["cate-idx"], airbench[i]["l2-name"], airbench[i]["l3-name"], airbench[i]["l4-name"], airbench[i]["prompt"]])
        question_list.append(airbench[i]["prompt"])
    
    result_list = batcher_answer.handle_message_list(question_list)

    model_string = model_string.replace('/','_')
    with open(f'answer_{model_string}_result.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'response'])

        for i, row in enumerate(row_list):
            row.append(result_list[i])
            writer.writerow(row)
