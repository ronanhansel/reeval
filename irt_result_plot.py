import pandas as pd
import matplotlib.pyplot as plt
# from tueplots import bundles
# bundles.icml2022()
# bundles.icml2022(family="sans-serif", usetex=False, column="full", nrows=2)
# plt.rcParams.update(bundles.icml2022())

if __name__ == "__main__":
    os.cmd("Rscript R/mirt.R")
    
    perturb_list = ["base", "perturb1", "perturb2"]
    model_list = ["1PL", "2PL","3PL"]

    for perturb in perturb_list:
        for model in model_list:
            file_path = f'model_coef/divided_{perturb}_coef_{model}.csv'
            df = pd.read_csv(file_path)

            # delete column
            df = df.iloc[:, 1:-2]

            new_columns = ['a1', 'd', 'g', 'u']
            data = {col: [] for col in new_columns}
            for i in range(0, len(df.columns), 4):
                for col, new_col in zip(df.columns[i:i+4], new_columns):
                    data[new_col].append(df[col].values[0])

            new_df = pd.DataFrame(data)
            new_df.to_csv(f'model_coef/divided_{perturb}_coef_{model}_clean.csv', index=False)

    for model_name in model_list:
        print(model_name)

        base_coef = pd.read_csv(f'model_coef/divided_base_coef_{model_name}_clean.csv', usecols=[0, 1, 2])
        perturb1_coef = pd.read_csv(f'model_coef/divided_perturb1_coef_{model_name}_clean.csv', usecols=[0, 1, 2])
        perturb2_coef = pd.read_csv(f'model_coef/divided_perturb2_coef_{model_name}_clean.csv', usecols=[0, 1, 2])

        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(base_coef.iloc[:, 0], base_coef.iloc[:, 1], base_coef.iloc[:, 2], c='r', label='Base Coef')
        ax.scatter(perturb1_coef.iloc[:, 0], perturb1_coef.iloc[:, 1], perturb1_coef.iloc[:, 2], c='g', label='Perturb1 Coef')
        ax.scatter(perturb2_coef.iloc[:, 0], perturb2_coef.iloc[:, 1], perturb2_coef.iloc[:, 2], c='b', label='Perturb2 Coef')

        ax.set_xlabel('a1')
        ax.set_ylabel('d')
        ax.set_zlabel('g')

        # ax.set_xlim(-3, 3)
        # ax.set_ylim(-3, 3)
        # ax.set_zlim(-3, 3)

        ax.legend()
        plt.show()

    for model_name in model_list:
        print(model_name)

        base_coef = pd.read_csv(f'model_coef/divided_base_coef_{model_name}_clean.csv', usecols=[0, 1, 2])
        perturb1_coef = pd.read_csv(f'model_coef/divided_perturb1_coef_{model_name}_clean.csv', usecols=[0, 1, 2])
        perturb2_coef = pd.read_csv(f'model_coef/divided_perturb2_coef_{model_name}_clean.csv', usecols=[0, 1, 2])

        base_value = [base_coef.iloc[:, i] for i in range(3)]
        perturb1_value = [perturb1_coef.iloc[:, i] for i in range(3)]
        perturb2_value = [perturb2_coef.iloc[:, i] for i in range(3)]

        plt.figure(figsize=(18, 18))

        def plot_hist(serial, data, color, para, perturb):
            plt.subplot(3, 3, serial)
            plt.hist(data, bins=30, color=color, density=True)
            plt.xlabel('value', fontsize=16)
            plt.ylabel('frequency', fontsize=16)
            plt.title(f'{para} of {perturb} coef', fontsize=18)
            plt.grid(True)

        plot_hist(1, base_value[0], 'r', 'a1', 'base')
        plot_hist(2, base_value[1], 'r', 'd', 'base')
        plot_hist(3, base_value[2], 'r', 'g', 'base')   

        plot_hist(4, perturb1_value[0], 'g', 'a1', 'perturb1')
        plot_hist(5, perturb1_value[1], 'g', 'd', 'perturb1')
        plot_hist(6, perturb1_value[2], 'g', 'g', 'perturb1')   

        plot_hist(7, perturb2_value[0], 'b', 'a1', 'perturb2')
        plot_hist(8, perturb2_value[1], 'b', 'd', 'perturb2')
        plot_hist(9, perturb2_value[2], 'b', 'g', 'perturb2')   

        plt.show()

