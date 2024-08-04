import os
from huggingface_hub import login
import datasets
from bacher_modify import GPTBatcher
from dotenv import load_dotenv
import csv
import argparse

def extract_content(tag, text):
    start_idx = text.find(tag)
    # if start_idx == -1:
    #     return None
    start_of_content = start_idx + len(tag)
    if tag == "##the_score: ":
        end_idx = text.find("\n", start_of_content)
    else:
        end_idx = text.find(".\n##", start_of_content)
    if end_idx == -1:
        content = text[start_of_content:].strip()
    else:
        content = text[start_of_content:end_idx].strip()
    return content

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--url', type=str, required=True, help='API base URL')
parser.add_argument('--start', type=int, required=True, help='Start index')
parser.add_argument('--end', type=int, required=True, help='End index')

args = parser.parse_args()
url = args.url
start = args.start
end = args.end

load_dotenv()
openai_key = os.getenv('OPENAI_KEY')
batcher_eval = GPTBatcher(api_key=openai_key,
                     model_name='gpt-4o',
                     temperature=0,
                     num_workers=64,
                    )

login(token = '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')
airbench = datasets.load_dataset("stanford-crfm/air-bench-2024", split="test")
judge_prompts = datasets.load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
persona = datasets.load_dataset("proj-persona/PersonaHub", "persona")

for j in range(start-1, end):
    system_prompt = "Assume you are the following person: " + persona['train'][j]["persona"]

    batcher_answer = GPTBatcher(api_key="EMPTY",
                        model_name='meta-llama/Meta-Llama-3-8B-Instruct',
                        temperature=0,
                        num_workers=64,
                        max_token=128,
                        system_prompt = system_prompt,
                        api_base_url=url,
                        )
    
    row_list = []
    question_list_1 = []
    result_list_1 = []
    for i in range(3000):
        row_list.append([airbench[i]["cate-idx"], airbench[i]["l2-name"], airbench[i]["l3-name"], airbench[i]["l4-name"], airbench[i]["prompt"]])
        question_list_1.append(airbench[i]["prompt"])
    
    result_list_1 = batcher_answer.handle_message_list(question_list_1)

    question_list_2 = []
    for i, row in enumerate(row_list):
        cate_idx = row[0]
        prompt = row[4]
        answer = result_list_1[i]
        row.append(answer)

        for data in judge_prompts:
            if data['cate-idx'] == cate_idx:
                judge_prompt = data['judge_prompt']
        
        question_list_2.append(judge_prompt.replace("{{QUESTION}}", prompt).replace("{{ANSWER}}", answer))

    result_list_2 = batcher_eval.handle_message_list(question_list_2)
    reason_list = [extract_content("##short_reasoning: ", result) for result in result_list_2]
    score_list = [float(extract_content("##the_score: ", result)) for result in result_list_2]

    with open(f'llama_3_8b_persona_{j}_result.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'response', 'score_reason', 'score'])

        for i, row in enumerate(row_list):
            cate_idx, l2_name, l3_name, l4_name, prompt, response = row
            reason = reason_list[i]
            score = score_list[i]

            writer.writerow([cate_idx, l2_name, l3_name, l4_name, prompt, response, reason, score])
