from datasets import Dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login
from dotenv import load_dotenv

def get_delete_index(input_dir):
    matrix_df = pd.DataFrame()
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            infile_path = os.path.join(input_dir, filename)
            data = pd.read_csv(infile_path)
            columns_to_keep = ['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'score']
            data = data[columns_to_keep]
            last_column = data.iloc[:, -1]
            model_name = filename.split(f"eval_")[1].split("_result.csv")[0]
            matrix_df[model_name] = last_column
            
    matrix_df = matrix_df.replace(0.5, 1)
    matrix_df = matrix_df.astype(int)
    delete_index = []
    for index, row in matrix_df.iterrows():
        if row.nunique() == 1:
            delete_index.append(index)
    return delete_index
            
if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)

    question_text_path = "/Users/tyhhh/Desktop/certified-eval/data/real/pre_irt_data/eval/eval_01-ai_yi-34b-chat_result.csv"
    Z_path = "/Users/tyhhh/Desktop/certified-eval/data/real/irt_result/Z/all_1PL_Z_clean.csv"
    matrix_path = "/Users/tyhhh/Desktop/certified-eval/data/real/response_matrix/all_matrix.csv"
    input_dir = "/Users/tyhhh/Desktop/certified-eval/data/real/pre_irt_data/eval"
    
    question_df = pd.read_csv(question_text_path, usecols=[4])
    Z_df = pd.read_csv(Z_path, usecols=[2])
    print(f"len(question_df): {len(question_df)}")
    print(f"len(Z_df): {len(Z_df)}")
    
    same_value_columns_indices = get_delete_index(input_dir)
    print(f"len(same_value_columns_indices): {len(same_value_columns_indices)}")
    
    question_df = question_df.drop(question_df.index[same_value_columns_indices])
    assert len(question_df) == len(Z_df), "The lengths of question_df and Z_df must be the same."

    combined_df = pd.concat([question_df, Z_df], axis=1)
    combined_df.columns = ['question_text', 'z3']
    hf_dataset = Dataset.from_pandas(combined_df)
    dataset_dict = DatasetDict({"train": hf_dataset})
    dataset_dict.push_to_hub("stair-lab/airbench-difficulty", private=True)

