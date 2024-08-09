from together import Together
from concurrent.futures import ThreadPoolExecutor, wait
from tqdm import tqdm

class TogetherBatcher:

    def __init__(self,api_key,model_name,system_prompt="",temperature=1,max_token=256,num_workers=64,timeout_duration=60,retry_attempts=2,api_base_url=None):
        
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_token = max_token
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index =[]
        if api_base_url:
            self.client.base_url = api_base_url

    def get_attitude(self, ask_text):
        index, ask_text = ask_text
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ask_text}
                ],
                temperature=self.temperature,
                max_tokens=self.max_token,
            )
            return (index, completion.choices[0].message.content)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.miss_index.append(index)
            return (index, None)

    def process_attitude(self, message_list):
        new_list = []
        num_workers = self.num_workers
        timeout_duration = self.timeout_duration
        retry_attempts = 2
    
        executor = ThreadPoolExecutor(max_workers=num_workers)
        message_chunks = list(self.chunk_list(message_list, num_workers))
        try:
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {executor.submit(self.get_attitude, message): message for message in chunk}
                for _ in range(retry_attempts):
                    done, not_done = wait(future_to_message.keys(), timeout=timeout_duration)
                    # Cancel all unfinished Future objects
                    for future in not_done:
                        future.cancel()
                    # Add the results of the completed Future objects to the new list
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    # Resubmit the tasks for all unfinished Future objects
                    future_to_message = {executor.submit(self.get_attitude, future_to_message[future]): future for future in not_done}
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            executor.shutdown(wait=False)
            return new_list

    def complete_attitude_list(self, attitude_list, max_length):
        completed_list = []
        current_index = 0
        for item in attitude_list:
            index, value = item
            # Fill in missing indices
            while current_index < index:
                completed_list.append((current_index, None))
                current_index += 1
            # Add the current element from the list
            completed_list.append(item)
            current_index = index + 1
        while current_index < max_length:
            print("Filling in missing index", current_index)
            self.miss_index.append(current_index)
            completed_list.append((current_index, None))
            current_index += 1
        return completed_list

    def chunk_list(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def handle_message_list(self, message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def get_miss_index(self):
        return self.miss_index

model_string_list = [
    # # "zero-one-ai/Yi-34B-Chat",
    # "Austism/chronos-hermes-13b",
    # "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    # # "databricks/dbrx-instruct",
    # "deepseek-ai/deepseek-coder-33b-instruct",
    # # "deepseek-ai/deepseek-llm-67b-chat",
    # "garage-bAInd/Platypus2-70B-instruct",
    # "google/gemma-2b-it",
    # "google/gemma-7b-it",
    # "Gryphe/MythoMax-L2-13b",
    # "lmsys/vicuna-13b-v1.5",
    # "lmsys/vicuna-7b-v1.5",
    # "codellama/CodeLlama-13b-Instruct-hf",
    # "codellama/CodeLlama-34b-Instruct-hf",
    # "codellama/CodeLlama-70b-Instruct-hf",
    # "codellama/CodeLlama-7b-Instruct-hf",
    # "meta-llama/Llama-2-70b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "meta-llama/Llama-2-7b-chat-hf",
    # # "meta-llama/Llama-3-8b-chat-hf",
    # # "meta-llama/Llama-3-70b-chat-hf",
    # "mistralai/Mistral-7B-Instruct-v0.1",
    # "mistralai/Mistral-7B-Instruct-v0.2",
    # # "mistralai/Mistral-7B-Instruct-v0.3",
    # # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # # "mistralai/Mixtral-8x22B-Instruct-v0.1",
    # "NousResearch/Nous-Capybara-7B-V1p9",
    # "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    # "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    # "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
    # "NousResearch/Nous-Hermes-llama-2-7b",
    # "NousResearch/Nous-Hermes-Llama2-13b",
    # "NousResearch/Nous-Hermes-2-Yi-34B",
    # "openchat/openchat-3.5-1210",
    # "Open-Orca/Mistral-7B-OpenOrca",
    # "Qwen/Qwen1.5-0.5B-Chat",
    # "Qwen/Qwen1.5-1.8B-Chat",
    # "Qwen/Qwen1.5-4B-Chat",
    # "Qwen/Qwen1.5-7B-Chat",
    # "Qwen/Qwen1.5-14B-Chat",
    # "Qwen/Qwen1.5-32B-Chat",
    # # "Qwen/Qwen1.5-72B-Chat",
    # "Qwen/Qwen1.5-110B-Chat",
    # "Qwen/Qwen2-72B-Instruct",
    # "snorkelai/Snorkel-Mistral-PairRM-DPO",
    # "Snowflake/snowflake-arctic-instruct",
    # "togethercomputer/alpaca-7b",
    "teknium/OpenHermes-2-Mistral-7B",
    "teknium/OpenHermes-2p5-Mistral-7B",
    "togethercomputer/Llama-2-7B-32K-Instruct",
    "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    "togethercomputer/RedPajama-INCITE-7B-Chat",
    "togethercomputer/StripedHyena-Nous-7B",
    "Undi95/ReMM-SLERP-L2-13B",
    "Undi95/Toppy-M-7B",
    "WizardLM/WizardLM-13B-V1.2",
    "upstage/SOLAR-10.7B-Instruct-v1.0"
]

print(len(model_string_list))

import csv
import datasets
import os
from dotenv import load_dotenv

test_data = datasets.load_dataset("stanford-crfm/air-bench-2024", "default", split="test")
print(test_data[612])
print(test_data[1910])
raw_data = test_data[612:1911]

load_dotenv()
together_key = os.getenv('TOGETHERAI_KEY')

for model_string in model_string_list:
    print(model_string)

    batcher = TogetherBatcher(api_key=together_key,
                     model_name=model_string,
                     temperature=0,
                     num_workers=64,
                     max_token=128,
                    )
    
    row_list = []
    question_list = []

    for i in range(len(raw_data['cate-idx'])):
        cate_idx = raw_data['cate-idx'][i]
        l2_name = raw_data['l2-name'][i]
        l3_name = raw_data['l3-name'][i]
        l4_name = raw_data['l4-name'][i]
        prompt = raw_data['prompt'][i]

        question_list.append(prompt)
        row_list.append([cate_idx, l2_name, l3_name, l4_name, prompt])

    result_list = batcher.handle_message_list(question_list)

    model_string_2 = model_string.replace("/", "_")

    with open(f'../raw_data_more/answer/{model_string_2}_result.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'response'])

        for i, row in enumerate(row_list):
            cate_idx, l2_name, l3_name, l4_name, prompt = row
            response = result_list[i]
            writer.writerow([cate_idx, l2_name, l3_name, l4_name, prompt, response])

