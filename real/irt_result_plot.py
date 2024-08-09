import pandas as pd
import matplotlib.pyplot as plt
import subprocess
# from tueplots import bundles
# bundles.icml2022()
# bundles.icml2022(family="sans-serif", usetex=False, column="full", nrows=2)
# plt.rcParams.update(bundles.icml2022())

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})

    # run mirt.R
    # subprocess.run("conda activate R", shell=True, check=True, executable="/bin/bash")
    # subprocess.run("Rscript mirt.R", shell=True, check=True)
    
    
    
    # clean irt_reslult/Z
    perturb_list = ["base", "perturb1", "perturb2"]
    model_list = ["1PL", "2PL","3PL"]

    for perturb in perturb_list:
        for model in model_list:
            df = pd.read_csv(f'../data/real/irt_reslult/Z/{perturb}_{model}_Z.csv')

            # delete column
            df = df.iloc[:, 1:-2]

            # overleaf/R_library: z1/g, z2/a1, z3/d
            new_columns = ['z2', 'z3', 'z1', 'u']
            data = {col: [] for col in new_columns}
            for i in range(0, len(df.columns), 4):
                for col, new_col in zip(df.columns[i:i+4], new_columns):
                    data[new_col].append(df[col].values[0])

            new_df = pd.DataFrame(data)
            new_df = new_df[['z1', 'z2', 'z3']]
            new_df.to_csv(f'../data/real/irt_reslult/theta/{perturb}_{model}_Z_clean.csv', index=False)



    # 3D plot of [z1, z2, z3], only 3PL
    base_coef = pd.read_csv(f'../data/real/irt_reslult/theta/base_3PL_Z_clean.csv')
    perturb1_coef = pd.read_csv(f'../data/real/irt_reslult/theta/perturb1_3PL_Z_clean.csv')
    perturb2_coef = pd.read_csv(f'../data/real/irt_reslult/theta/perturb2_3PL_Z_clean.csv')

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(base_coef.iloc[:, 0], base_coef.iloc[:, 1], base_coef.iloc[:, 2], c='r', label='Base Z')
    ax.scatter(perturb1_coef.iloc[:, 0], perturb1_coef.iloc[:, 1], perturb1_coef.iloc[:, 2], c='g', label='Perturb1 Z')
    ax.scatter(perturb2_coef.iloc[:, 0], perturb2_coef.iloc[:, 1], perturb2_coef.iloc[:, 2], c='b', label='Perturb2 Z')

    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    ax.set_zlabel(r'$z_3$')

    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-3, 3)
    # ax.set_zlim(-3, 3)

    ax.legend()
    plt.savefig('/Users/tyhhh/Desktop/certified-eval/plot/real/real_3d3pl.png')
    plt.close(fig)



    # # plot Z distribution (density histogram)
    # for model in model_list:
    #     base_coef = pd.read_csv(f'../data/real/irt_reslult/theta/base_{model}_Z_clean.csv')
    #     perturb1_coef = pd.read_csv(f'model_coef/divided_perturb1_coef_{model}_clean.csv', usecols=[0, 1, 2])
    #     perturb2_coef = pd.read_csv(f'model_coef/divided_perturb2_coef_{model}_clean.csv', usecols=[0, 1, 2])

    #     base_value = [base_coef.iloc[:, i] for i in range(3)]
    #     perturb1_value = [perturb1_coef.iloc[:, i] for i in range(3)]
    #     perturb2_value = [perturb2_coef.iloc[:, i] for i in range(3)]

    #     plt.figure(figsize=(18, 18))

    #     def plot_hist(serial, data, color, para, perturb):
    #         plt.subplot(3, 3, serial)
    #         plt.hist(data, bins=30, color=color, density=True)
    #         plt.xlabel('value', fontsize=16)
    #         plt.ylabel('frequency', fontsize=16)
    #         plt.title(f'{para} of {perturb} coef', fontsize=18)
    #         plt.grid(True)

    #     plot_hist(1, base_value[0], 'r', 'a1', 'base')
    #     plot_hist(2, base_value[1], 'r', 'd', 'base')
    #     plot_hist(3, base_value[2], 'r', 'g', 'base')   

    #     plot_hist(4, perturb1_value[0], 'g', 'a1', 'perturb1')
    #     plot_hist(5, perturb1_value[1], 'g', 'd', 'perturb1')
    #     plot_hist(6, perturb1_value[2], 'g', 'g', 'perturb1')   

    #     plot_hist(7, perturb2_value[0], 'b', 'a1', 'perturb2')
    #     plot_hist(8, perturb2_value[1], 'b', 'd', 'perturb2')
    #     plot_hist(9, perturb2_value[2], 'b', 'g', 'perturb2')   

    #     plt.show()

