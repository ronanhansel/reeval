import re
import argparse
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
import pickle
from vllm import LLM

scenario2pattern = {
    "mmlu": r"(Question:)"
    # TODO: this pattern do not work for all scenarios, but split by '\n\n' is also problemetic, should write a scenario2pattern config
}


mmlu_context = """
    The following is a multiple choice (A, B, C, or D) question about %s from the Massive Multitask Language Understanding (MMLU) benchmark, 
    designed to measure ability in knowledge-intensive question answering across 57 domains. 
"""

def extract_last_question(pattern, text):
    # Split into individual questions using 'Question:' as the delimiter
    questions = re.split(pattern, text)
  
    # Remove empty strings and strip whitespace
    questions = [q.strip() for q in questions if q.strip()]
    
    # Get the last question (ignoring the last empty answer)
    last_question = questions[-1]
    
    # Remove the 'Answer:' part if it exists
    last_question = re.split(r'Answer:', last_question)[0].strip()
    
    return last_question

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)  # mmlu/mmlu
    args = parser.parse_args()

    repo_id = "stair-lab/reeval_csv"
    data_folder = snapshot_download(repo_id=repo_id, repo_type="dataset")
    data = pickle.load(open(f"{data_folder}/{args.dataset}/responses.pkl", "rb"))
    question_keys = pd.read_csv(f"{data_folder}/{args.dataset}/instances.csv")
    model_key = pd.read_csv(f"{data_folder}/model_df.csv")
    flop = pd.read_csv(f"{data_folder}/FLOP.csv")
    # For model_names_reeval, replace("_", "/") 
    flop["model_names_reeval"] = flop["model_names_reeval"].str.replace("_", "/")

    # merge model_key with flop: `model_names_reeval` should match `name` in model_key
    # not every name in model_key is in flop, and not every name in flop is in model_key
    # the resulting dataframe should has the same length as model_key
    model_key_ = pd.merge(model_key, flop, left_on="name", right_on="model_names_reeval", how="left")

    # remove the few-shot example from prompt
    pattern = scenario2pattern[args.dataset]
    n_questions = len(question_keys["prompt"])
    last_questions = [extract_last_question(pattern, question_keys["prompt"][i]) for i in range(n_questions)]
    question_keys["question_content"] = last_questions

    question_contexts = []
    for i in range(n_questions):
        subject_i = question_keys["subject"][i]
        subject_i = subject_i.replace("_", " ")
        question_contexts.append(mmlu_context % subject_i)
    question_keys["question_contexts"] = question_contexts

    # merge data and instances by instance_id
    # instance_id in data is float, while in instances is int
    # but they are the same
    data["instance_id"] = data["instance_id"].astype(int)
    data = pd.merge(data, question_keys, on="instance_id")

    # merge data with model_key by model_id
    data["model_id"] = data["model_id"].astype(int)
    data = pd.merge(data, model_key, on="model_id")   

    # save the data to parquet form
    data.to_parquet("data.parquet")

    # add a column for embedding (each embedding is a list of 4096 floats)
    content_to_embed = [context + question for context, question in zip(question_contexts, last_questions)]
    model = LLM(model="intfloat/e5-mistral-7b-instruct", enforce_eager=True)
    embeddings = model.encode(content_to_embed)
    embeddings = [emb.outputs.embedding for emb in embeddings]
    question_keys["embedding"] = embeddings
    question_keys.to_parquet("question.parquet")

    # upload the data to huggingface hub
    api = HfApi()
    api.create_repo("stair-lab/reeval_another_repo", repo_type="dataset", exist_ok=True)
    api.upload_file(
        repo_id="stair-lab/reeval_another_repo",
        path_in_repo="mmlu/data.parquet",
        path_or_fileobj="data.parquet",
        repo_type="dataset"
    )
    api.upload_file(
        repo_id="stair-lab/reeval_another_repo",
        path_in_repo="mmlu/question.parquet",
        path_or_fileobj="question.parquet",
        repo_type="dataset"
    )

    # load the data from huggingface
    data_folder = snapshot_download(repo_id="stair-lab/reeval_another_repo", repo_type="dataset")
    data = pd.read_parquet(f"{data_folder}/mmlu/data.parquet", engine="fastparquet")
    question_keys = pd.read_parquet(f"{data_folder}/mmlu/question.parquet", engine="fastparquet")
