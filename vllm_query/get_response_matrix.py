import pandas as pd
import os

if __name__ == "__main__":
    input_dir = "../data/real/pre_irt_data/eval"
    perturb_list = ["base", "perturb1", "perturb2"]
    i = 0

    for perturb in perturb_list:
        matrix_df = pd.DataFrame()

        for filename in os.listdir(input_dir):
            if filename.endswith(".csv"):
                infile_path = os.path.join(input_dir, filename)
                data = pd.read_csv(infile_path)
                columns_to_keep = ['cate-idx', 'l2-name', 'l3-name', 'l4-name', 'prompt', 'score']
                data = data[columns_to_keep]
                
                # Select every 3rd row
                data = data.iloc[i::3, :]

                # Extract the last column which is the score
                last_column = data.iloc[:, -1]
                model_name = filename.split(f"eval_")[1].split("_result.csv")[0]
                matrix_df[model_name] = last_column

        matrix_df = matrix_df.replace(0.5, 1)
        matrix_df = matrix_df.astype(int)
        matrix_df = matrix_df.reindex(sorted(matrix_df.columns), axis=1)
        matrix_df.reset_index(drop=True, inplace=True)
        matrix_df = matrix_df.T
        
        matrix_df.to_csv(f'../data/real/response_matrix/{perturb}_matrix.csv')
        i += 1
