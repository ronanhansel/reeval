from vllm import LLM, SamplingParams
import os
from huggingface_hub import login
import datasets
from gpt_batch.batcher import GPTBatcher
from dotenv import load_dotenv
import csv

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

start = 1
end = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

load_dotenv()
openai_key = os.getenv('OPENAI_KEY')
batcher = GPTBatcher(api_key=openai_key,
                     model_name='gpt-4o',
                     temperature=0,
                     num_workers=64,
                    )

login(token = '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')
airbench = datasets.load_dataset("stanford-crfm/air-bench-2024", split="test")
judge_prompts = datasets.load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
persona = datasets.load_dataset("proj-persona/PersonaHub", "persona")

for j in range(start-1, end):
    system_prompt = persona['train'][j]["persona"]

    prompts = []
    row_list = []
    for i in range(3000):
        row_list.append([airbench[i]["cate-idx"], airbench[i]["l2-name"], airbench[i]["l3-name"], airbench[i]["l4-name"]])
        prompts.append(airbench[i]["prompt"])

    # sampling_params = SamplingParams(temperature=0, max_tokens=128)
    # llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    # outputs = llm.generate(prompts, sampling_params)

    # question_list = []
    # row_list_2 = []
    # for i, row in enumerate(row_list):
    #     cate_idx = row[0]
    #     prompt = outputs[i].prompt
    #     answer = outputs[i].outputs[0].text
    #     row_list_2.append([prompt, answer])

    #     for data in judge_prompts:
    #         if data['cate-idx'] == cate_idx:
    #             judge_prompt = data['judge_prompt']
        
    #     question_list.append(judge_prompt.replace("{{QUESTION}}", prompt).replace("{{ANSWER}}", answer))

    # result_list = batcher.handle_message_list(question_list)
    # reason_list = [extract_content("##short_reasoning: ", result) for result in result_list]
    # score_list = [float(extract_content("##the_score: ", result)) for result in result_list]

    # with open(f'result.csv', 'w', newline='', encoding='utf-8') as outfile:
    #     writer = csv.writer(outfile)
    #     writer.writerow(['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'response','score_reason', 'score'])

    #     for i, row in enumerate(row_list):
    #         cate_idx, l2_name, l3_name, l4_name = row
    #         prompt = row_list_2[i][0]
    #         response = row_list_2[i][1]
    #         reason = reason_list[i]
    #         score = score_list[i]

    #         writer.writerow([cate_idx, l2_name, l3_name, l4_name, prompt, response, reason, score])
