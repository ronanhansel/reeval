import matplotlib.pyplot as plt
import jax.numpy as jnp
from utils import load_state
import os

if __name__ == '__main__':
    state_dir = "../data/synthetic/CAT"
    question_num = 10000
    theta_star = 1.25
    strategy_list = ['random', 'fisher', 'owen', 'modern']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, strategy in enumerate(strategy_list):
        state_path = os.path.join(state_dir, f"{strategy}_{question_num}.pt")
        state = load_state(state_path)
        
        if state:
            z3 = state['z3']
            theta_means = state['theta_means']
            theta_stds = state['theta_stds']
            
            total_question_nums = range(question_num)
            subset_question_num = len(theta_means)
            subset_question_nums = range(subset_question_num)

            axs[0, i].plot(
                subset_question_nums, 
                [theta_star] * subset_question_num, 
                label='True Theta', 
                color='black', 
                linestyle='--'
                )
            axs[0, i].plot(subset_question_nums, theta_means, label=f'{strategy}')
            theta_means = jnp.array(theta_means)
            theta_stds = jnp.array(theta_stds)
            axs[0, i].fill_between(subset_question_nums, 
                                theta_means - 3 * theta_stds, 
                                theta_means + 3 * theta_stds, 
                                alpha=0.2)
            axs[0, i].set_title(f'{strategy}')
            axs[0, i].set_xlabel('Number of Questions')
            axs[0, i].set_ylabel('Theta')
            axs[0, i].set_ylim([-4, 4])
            axs[0, i].grid(True)
            axs[0, i].legend()

    plt.tight_layout()
    plt.savefig('../plot/synthetic/random_adaptive_test_subplot.png')
