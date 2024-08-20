import gc
import sys
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import torch
import random
from synthetic_testtaker import SimulatedTestTaker
from fit_theta import fit_theta_mcmc
import jax.numpy as jnp
from utils import item_response_fn_1PL, clear_caches
import numpy as np

# export JAX_PLATFORM_NAME=cpu
# nohup python CAT.py > /scratch/yuhengtu/workspace/certified-eval/src/CAT.log 2>&1 &
  
def CAT_owen(z3, unasked_question_list, theta_mean):
    z3_unasked = z3[unasked_question_list]
    z3_unasked = jnp.array(z3_unasked)
    diff = jnp.abs(z3_unasked - theta_mean)
    return unasked_question_list[jnp.argmin(diff)]

def CAT_fisher(z3, unasked_question_list, theta_mean):
    fisher_info_list = []
    for unasked_question_index in unasked_question_list:
        theta = torch.tensor(theta_mean.item(), requires_grad=True)
        z_single = z3[unasked_question_index].clone().detach()    
        prob = item_response_fn_1PL(z_single, theta)
        hessian = prob * (1 - prob)
        fisher_info_list.append(hessian)
    index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
    return unasked_question_list[index_with_max_fisher_info]

def CAT_modern(z3, unasked_question_list, theta_samples):
    theta_samples_tensor = torch.tensor(np.array(theta_samples))
    fisher_info_list = []
    for unasked_question_index in unasked_question_list:
        z_single = z3[unasked_question_index].clone().detach()
        z_single_expanded = z_single.expand(theta_samples_tensor.shape[0])
        prob = item_response_fn_1PL(z_single_expanded, theta_samples_tensor)
        hessian = prob * (1 - prob)
        fisher_info_list.append(hessian.mean())
    unasked_question_index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
    return unasked_question_list[unasked_question_index_with_max_fisher_info]
    
def main(z3, new_testtaker, strategy, subset_question_num):
    print(f'strategy: {strategy}')

    question_num = len(z3)

    init_question_index = random.randint(0, question_num - 1)
    init_answer = new_testtaker.ask(z3, init_question_index)

    unasked_question_list = [i for i in range(question_num)]
    unasked_question_list.remove(init_question_index)
    asked_question_list = [init_question_index]
    asked_answer_list = [init_answer.float()]

    theta_means = []
    theta_stds = []
    for epoch in range(subset_question_num):
        print(f'\nepoch: {epoch+1}')
        asked_question_jnp = jnp.array(asked_question_list)
        asked_answer_jnp = jnp.array(asked_answer_list)
        z3_jnp = jnp.array(z3)

        mean_theta, std_theta, theta_samples = fit_theta_mcmc(
            z3_jnp, 
            asked_question_jnp, 
            asked_answer_jnp
            )
        theta_means.append(mean_theta)
        theta_stds.append(std_theta)

        if len(unasked_question_list) == 0:
            break
        
        if strategy=="random":
            new_question_index = random.choice(unasked_question_list)
        elif strategy=="owen":
            new_question_index = CAT_owen(z3, unasked_question_list, mean_theta)
        elif strategy=="fisher":
            new_question_index = CAT_fisher(z3, unasked_question_list, mean_theta)
        elif strategy=="modern":
            new_question_index = CAT_modern(z3, unasked_question_list, theta_samples)
        else:
            raise ValueError("strategy not supported")
        
        new_answer = new_testtaker.ask(z3, new_question_index)
        unasked_question_list.remove(new_question_index)
        asked_question_list.append(new_question_index)
        asked_answer_list.append(new_answer.float())
        
        clear_caches()
        
    return theta_means, theta_stds

if __name__ == "__main__":
    torch.manual_seed(10)
    
    # df = pd.read_csv(
    #     '../data/synthetic/irt_result/Z/synthetic_1PL_Z_clean.csv'
    #     )
    # z3 = torch.tensor(df.iloc[:, -1].tolist())
    # question_num = len(z3)
    
    question_num = 10000
    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    
    print(f'num of total questions: {question_num}')
    
    new_testtaker = SimulatedTestTaker(theta=1.25, model="1PL")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")

    subset_question_num = 1000
    random_theta_means, random_theta_stds = main(z3, new_testtaker, strategy="random", subset_question_num=subset_question_num)
    owen_theta_means, owen_theta_stds = main(z3, new_testtaker, strategy="owen", subset_question_num=subset_question_num)
    fisher_theta_means, fisher_theta_stds = main(z3, new_testtaker, strategy="fisher", subset_question_num=subset_question_num)
    modern_theta_means, modern_theta_stds = main(z3, new_testtaker, strategy="modern", subset_question_num=subset_question_num)
    
    total_question_nums = range(question_num)
    subset_question_nums = range(subset_question_num)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # First subplot - Random Testing
    axs[0, 0].plot(subset_question_nums, [theta_star] * subset_question_num, label='True Theta', color='black', linestyle='--')
    axs[0, 0].plot(subset_question_nums, random_theta_means, label='Random Testing', color='blue')
    random_theta_means = jnp.array(random_theta_means)
    random_theta_stds = jnp.array(random_theta_stds)
    axs[0, 0].fill_between(subset_question_nums, 
                        random_theta_means - 3 * random_theta_stds, 
                        random_theta_means + 3 * random_theta_stds, 
                        color='blue', 
                        alpha=0.2)
    axs[0, 0].set_title('Random Testing')
    axs[0, 0].set_ylim([-4, 4])
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Second subplot - CAT with Fisher
    axs[0, 1].plot(subset_question_nums, [theta_star] * subset_question_num, label='True Theta', color='black', linestyle='--')
    axs[0, 1].plot(subset_question_nums, fisher_theta_means, label='CAT with fisher', color='red')
    fisher_theta_means = jnp.array(fisher_theta_means)
    fisher_theta_stds = jnp.array(fisher_theta_stds)
    axs[0, 1].fill_between(subset_question_nums, 
                        fisher_theta_means - 3 * fisher_theta_stds, 
                        fisher_theta_means + 3 * fisher_theta_stds, 
                        color='red', 
                        alpha=0.2)
    axs[0, 1].set_title('CAT with Fisher')
    axs[0, 1].set_ylim([-4, 4])
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Third subplot - CAT with Owen
    axs[1, 0].plot(subset_question_nums, [theta_star] * subset_question_num, label='True Theta', color='black', linestyle='--')
    axs[1, 0].plot(subset_question_nums, owen_theta_means, label='CAT with owen', color='green')
    owen_theta_means = jnp.array(owen_theta_means)
    owen_theta_stds = jnp.array(owen_theta_stds)
    axs[1, 0].fill_between(subset_question_nums, 
                           owen_theta_means - 3 * owen_theta_stds, 
                           owen_theta_means + 3 * owen_theta_stds, 
                           color='green', 
                           alpha=0.2)
    axs[1, 0].set_title('CAT with Owen')
    axs[1, 0].set_ylim([-4, 4])
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Fourth subplot - CAT with Modern
    axs[1, 1].plot(subset_question_nums, [theta_star] * subset_question_num, label='True Theta', color='black', linestyle='--')
    axs[1, 1].plot(subset_question_nums, modern_theta_means, label='CAT with modern', color='purple')
    modern_theta_means = jnp.array(modern_theta_means)
    modern_theta_stds = jnp.array(modern_theta_stds)
    axs[1, 1].fill_between(subset_question_nums, 
                           modern_theta_means - 3 * modern_theta_stds, 
                           modern_theta_means + 3 * modern_theta_stds, 
                           color='purple', 
                           alpha=0.2)
    axs[1, 1].set_title('CAT with Modern')
    axs[1, 1].set_ylim([-4, 4])
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # Adjusting the layout and saving the figure
    for ax in axs.flat:
        ax.set_xlabel('Number of Questions')
        ax.set_ylabel('Theta')

    plt.tight_layout()
    plt.savefig('../plot/synthetic/random_adaptive_test_subplot.png')

    
    print("finish!!!")