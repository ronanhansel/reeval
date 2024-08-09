import pandas as pd
import os

input_dir_1 = "./raw_data"
input_dir_2 = "./raw_data_more"

perturb_list = ["base", "perturb1", "perturb2"]
i = 0

for perturb in perturb_list:
    output_dir = f"./clean_data/{perturb}"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir_1):
        if filename.endswith(".csv"):
            infile_path = os.path.join(input_dir_1, filename)
            outfile_path = os.path.join(output_dir, f"{perturb}_{filename}")
            
            data = pd.read_csv(infile_path)

            # keep some columns
            columns_to_keep = ['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'score']
            data = data[columns_to_keep]
            
            # get selected rows (every 3 row)
            data = data.iloc[i+612:1911:3, :]
            
            # add one column: prompt_idx
            data.insert(0, 'prompt_idx', range(0, len(data)))

            data.to_csv(outfile_path, index=False)

    for filename in os.listdir(input_dir_2):
        if filename.endswith(".csv"):
            infile_path = os.path.join(input_dir_2, filename)
            outfile_path = os.path.join(output_dir, f"{perturb}_{filename}")
            
            data = pd.read_csv(infile_path)

            # keep some columns
            columns_to_keep = ['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'score']
            data = data[columns_to_keep]
            
            # get selected rows (every 3 row)
            data = data.iloc[i::3, :]
            
            # add one column: prompt_idx
            data.insert(0, 'prompt_idx', range(0, len(data)))

            data.to_csv(outfile_path, index=False)

    i += 1

def format_model_name(model):
    if model == '01-ai_yi-34b-chat':
        return 'Yi Chat (34B)'
    elif model == 'anthropic_claude-3-haiku-20240307':
        return 'Claude 3 Haiku'
    elif model == 'anthropic_claude-3-opus-20240229':
        return 'Claude 3 Opus'
    elif model == 'anthropic_claude-3-sonnet-20240229':
        return 'Claude 3 Sonnet'
    elif model == 'cohere_command-r-plus':
        return 'Cohere Command R Plus'
    elif model == 'cohere_command-r':
        return 'Cohere Command R'
    elif model == 'databricks_dbrx-instruct':
        return 'DBRX Instruct'
    elif model == 'deepseek-ai_deepseek-llm-67b-chat':
        return 'DeepSeek LLM Chat (67B)'
    elif model == 'google_gemini-1.5-flash-001-safety-block-none':
        return 'Gemini 1.5 Flash'
    elif model == 'google_gemini-1.5-pro-001-safety-block-none':
        return 'Gemini 1.5 Pro'
    elif model == 'meta_llama-3-8b-chat':
        return 'Llama 3 Instruct (8B)'
    elif model == 'meta_llama-3-70b-chat':
        return 'Llama 3 Instruct (70B)'
    elif model == 'mistralai_mistral-7b-instruct-v0.3':
        return 'Mistral Instruct v0.3 (7B)'
    elif model == 'mistralai_mixtral-8x7b-instruct-v0.1':
        return 'Mixtral Instruct (8x7B)'
    elif model == 'mistralai_mixtral-8x22b-instruct-v0.1':
        return 'Mixtral Instruct (8x22B)'
    elif model == 'openai_gpt-3.5-turbo-0613':
        return 'GPT-3.5 Turbo (0613)'
    elif model == 'openai_gpt-3.5-turbo-0125':
        return 'GPT-3.5 Turbo (0125)'
    elif model == 'openai_gpt-3.5-turbo-1106':
        return 'GPT-3.5 Turbo (1106)'
    elif model == 'openai_gpt-4-turbo-2024-04-09':
        return 'GPT-4 Turbo'
    elif model == 'openai_gpt-4o-2024-05-13':
        return 'GPT-4o'
    elif model == 'qwen_qwen1.5-72b-chat':
        return 'Qwen1.5 Chat (72B)'
    elif model == 'openai_gpt-3.5-turbo-0301':
        return 'GPT-3.5 Turbo (0301)'
    else:
        return model

perturb_list = ["base", "perturb1", "perturb2"]
for perturb in perturb_list:
    # initialize
    matrix_df = pd.DataFrame()
    input_dir = f"./clean_data/{perturb}"

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            infile_path = os.path.join(input_dir, filename)
            df = pd.read_csv(infile_path)
            last_column = df.iloc[:, -1]

            model_string = filename.split(f"{perturb}_")[1].split("_result.csv")[0]
            model_name = format_model_name(model_string)
            matrix_df[model_name] = last_column

    matrix_df = matrix_df.replace(0.5, 1)
    matrix_df = matrix_df.astype(int)

    matrix_df = matrix_df.reindex(sorted(matrix_df.columns), axis=1)
    matrix_df = matrix_df.T
    matrix_df.to_csv(f'./clean_data/{perturb}_matrix.csv')

# # divide testtakers

import pandas as pd
import os

df = pd.read_csv('clean_data/base/base_01-ai_yi-34b-chat_result.csv')

grouped_data = df.groupby('cate-idx')
group_counts = grouped_data.size()
min_count = group_counts.min()
print(min_count)


perturb_list = ["base", "perturb1", "perturb2"]
for perturb in perturb_list:
    # initialize
    matrix_df = pd.DataFrame()
    input_dir = f"clean_data/{perturb}"
    output_dir = f'clean_data/divided_{perturb}'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            model_string = filename.split(f"{perturb}_")[1].split("_result.csv")[0]

            infile_path = os.path.join(input_dir, filename)
            df = pd.read_csv(infile_path)

            grouped_data = df.groupby('cate-idx')
            selected_rows = [group_content.head(min_count) for group_name, group_content in grouped_data]
            selected_df = pd.concat(selected_rows)

            for i in range(min_count):
                split_df = pd.DataFrame()
                for group_name, group_content in grouped_data:
                    split_df = pd.concat([split_df, group_content.iloc[[i]]])
                split_df.to_csv(f'clean_data/divided_{perturb}/divided_{perturb}_{model_string}_{i}.csv', index=False)

perturb_list = ["base", "perturb1", "perturb2"]
for perturb in perturb_list:
    # initialize
    matrix_df = pd.DataFrame()
    input_dir = f"clean_data/divided_{perturb}"

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            infile_path = os.path.join(input_dir, filename)
            df = pd.read_csv(infile_path)
            last_column = df.iloc[:, -1]

            model_string = filename.split(f"divided_{perturb}_")[1].split(".csv")[0]
            model_serial = model_string.split("_")[-1]
            model_name_string = model_string.split(f"_{model_serial}")[0]

            model_name = format_model_name(model_name_string)

            matrix_df[f"{model_name} taker {model_serial}"] = last_column

    matrix_df = matrix_df.replace(0.5, 1)
    matrix_df = matrix_df.astype(int)

    matrix_df = matrix_df.reindex(sorted(matrix_df.columns), axis=1)
    matrix_df = matrix_df.T
    matrix_df.to_csv(f'./clean_data/divided_{perturb}_matrix.csv')

