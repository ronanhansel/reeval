import torch
from fit_theta import fit_theta_mcmc
from synthetic_testtaker import SimulatedTestTaker
import jax.numpy as jnp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.manual_seed(10)
    
    new_testtaker = SimulatedTestTaker(theta=1.25, model="1PL")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")
    
    max_question_num = 50000
    z3 = torch.normal(mean=0.0, std=1.0, size=(max_question_num,))
    asked_question_list = list(range(max_question_num))
    asked_answer_list = []
    for i in range(max_question_num):
        asked_answer_list.append(new_testtaker.ask(z3, i))
    
    theta_means = []
    theta_stds = []
    question_nums = range(200, max_question_num+1, 200)
    for question_num in question_nums:
        print(f'Question Num: {question_num}')
        asked_question_list_subset = jnp.array(asked_question_list[:question_num])
        asked_answer_list_subset = jnp.array(asked_answer_list[:question_num])
        z3_subset = jnp.array(z3[:question_num])

        mean_theta, std_theta, _ = fit_theta_mcmc(
            z3_subset, 
            asked_question_list_subset, 
            asked_answer_list_subset
            )
        theta_means.append(mean_theta)
        theta_stds.append(std_theta)

    plt.figure(figsize=(10, 6))
    plt.plot(question_nums, [theta_star] * len(question_nums), label='True Theta', color='black', linestyle='--')
    plt.plot(question_nums, theta_means, label='MCMC Theta Mean', color='blue')
    theta_means = jnp.array(theta_means)
    theta_stds = jnp.array(theta_stds)
    plt.fill_between(question_nums, theta_means - 3 * theta_stds, theta_means + 3 * theta_stds, color='blue', alpha=0.2)
    
    plt.xlabel('Number of Questions')
    plt.ylabel('Theta')
    plt.title('Theta Estimation vs. Number of Questions')
    plt.legend()
    plt.grid(True)
    plt.savefig('../plot/synthetic/fit_theta.png')