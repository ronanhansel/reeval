import csv
from gpt_batch.batcher import GPTBatcher
import datasets
import os
from dotenv import load_dotenv

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

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

batcher = GPTBatcher(api_key=openai_key,
                     model_name='gpt-4o',
                     temperature=0,
                     num_workers=64,
                    )

judge_prompt_list = datasets.load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
input_dir = "./raw_data_more/answer"
output_dir = f"./raw_data_more"

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        print(filename)
        infile_path = os.path.join(input_dir, filename)
        outfile_path = os.path.join(output_dir, filename)
    
        with open(infile_path, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            next(reader) # skip first row

            question_list = []
            row_list = []

            for i, row in enumerate(reader):
                cate_idx, l2_name, l3_name, l4_name, prompt, response = row

                # find corresponding judge_prompt
                for data in judge_prompt_list:
                    if data['cate-idx'] == cate_idx:
                        judge_prompt = data['judge_prompt']

                row_list.append([cate_idx, l2_name, l3_name, l4_name, prompt, response, judge_prompt])
                question_list.append(judge_prompt.replace("{{QUESTION}}", prompt).replace("{{ANSWER}}", response))

        # print(question_list[0])
        result_list = batcher.handle_message_list(question_list)

        reason_list = [extract_content("##short_reasoning: ", result) for result in result_list]
        score_list = [float(extract_content("##the_score: ", result)) for result in result_list]

        with open(outfile_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'response','judge_prompt','score_reason', 'score'])

            for i, row in enumerate(row_list):
                cate_idx, l2_name, l3_name, l4_name, prompt, response, judge_prompt = row
                reason = reason_list[i]
                score = score_list[i]

                writer.writerow([cate_idx, l2_name, l3_name, l4_name, prompt, response, judge_prompt, reason, score])
        

