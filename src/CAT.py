import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
from synthetic_testtaker import SimulatedTestTaker
from fit_theta import fit_theta_mcmc
import jax.numpy as jnp
from utils import item_response_fn_1PL

def choose_z_similar_to_theta(z3, unasked_question_list, theta_mean):
    z3_unasked = z3[unasked_question_list]
    z3_unasked = jnp.array(z3_unasked)
    diff = jnp.abs(z3_unasked - theta_mean)
    return unasked_question_list[jnp.argmin(diff)]

# def computarized_adaptive_testing(z3, unasked_question_list, theta_mean):
#     fisher_info_list = []
#     for unasked_question_index in unasked_question_list:
#         prob = item_response_fn_1PL(z3[unasked_question_index], theta_mean, datatype="jnp")
#         bernoulli = torch.distributions.Bernoulli(prob)
#         samples_Y = bernoulli.sample((100,))
#         hessian_list = [] # TODO: all the same

#         for sample in samples_Y:
#             prob = item_response_fn_1PL(z3[unasked_question_index], theta_mean, datatype="jnp")
#             bernoulli = torch.distributions.Bernoulli(prob)
#             log_prob = bernoulli.log_prob(sample)

#             grad = torch.autograd.grad(log_prob, theta_hat, create_graph=True)[0]
#             hessian = torch.autograd.grad(grad, theta_hat)[0]
#             hessian_list.append(hessian)

#         fisher_info = torch.stack(hessian_list).mean()
#         fisher_info_list.append(fisher_info)

#     unasked_question_index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
#     new_question_index = unasked_question_list[unasked_question_index_with_max_fisher_info]
#     print(f'new_question_index: {new_question_index}')

    
def main(z3, new_testtaker, strategy="random", subset_question_num=None):
    print(f'strategy: {strategy}')

    question_num = len(z3)
    if subset_question_num==None:
        subset_question_num = question_num

    init_question_index = random.randint(0, subset_question_num - 1)
    init_answer = new_testtaker.ask(z3, init_question_index)

    unasked_question_list = [i for i in range(question_num)]
    unasked_question_list.remove(init_question_index)
    asked_question_list = [init_question_index]
    asked_answer_list = [init_answer.float()]
    asked_z3_list = [z3[init_question_index]]

    theta_means = []
    theta_stds = []
    for epoch in range(subset_question_num):
        print(f'\nepoch: {epoch+1}')
        asked_question_jnp = jnp.array(asked_question_list)
        asked_answer_jnp = jnp.array(asked_answer_list)
        asked_z3_jnp = jnp.array(asked_z3_list)
        
        mean_theta, std_theta = fit_theta_mcmc(
            asked_z3_jnp, 
            asked_question_jnp, 
            asked_answer_jnp
            )
        theta_means.append(mean_theta)
        theta_stds.append(std_theta)

        if len(unasked_question_list) == 0:
            break
        
        if strategy=="random":
            new_question_index = random.choice(unasked_question_list)
        elif strategy=="similar":
            new_question_index = choose_z_similar_to_theta(z3, unasked_question_list, mean_theta)

        new_answer = new_testtaker.ask(z3, new_question_index)
        unasked_question_list.remove(new_question_index)
        asked_question_list.append(new_question_index)
        asked_answer_list.append(new_answer.float())
        asked_z3_list.append(z3[new_question_index])
        
    return theta_means, theta_stds

if __name__ == "__main__":
    torch.manual_seed(42)
    
    df = pd.read_csv(
        '../data/synthetic/irt_result/Z/synthetic_1PL_Z_clean.csv'
        )
    z3 = torch.tensor(df.iloc[:, -1].tolist())
    question_num = len(z3)
    
    print(f'num of total questions: {question_num}')
    
    new_testtaker = SimulatedTestTaker(theta=1.25, model="1PL")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")

    random_theta_means, random_theta_stds = main(z3, new_testtaker, strategy="random")
    
    subset_question_num = 100
    similar_theta_means, similar_theta_stds = main(z3, new_testtaker, strategy="similar", subset_question_num=subset_question_num)
    
    total_question_nums = range(question_num)
    subset_question_nums = range(subset_question_num)
    
    plt.figure(figsize=(10, 6))
    plt.plot(total_question_nums, [theta_star] * question_num, label='True Theta', color='black', linestyle='--')
    
    plt.plot(total_question_nums, random_theta_means, label='Random Testing', color='blue')
    random_theta_means = jnp.array(random_theta_means)
    random_theta_stds = jnp.array(random_theta_stds)
    plt.fill_between(total_question_nums, random_theta_means - 3 * random_theta_stds, random_theta_means + 3 * random_theta_stds, color='blue', alpha=0.2)
    
    plt.plot(subset_question_nums, similar_theta_means, label='choose z similar to theta', color='green')
    similar_theta_means = jnp.array(similar_theta_means)
    similar_theta_stds = jnp.array(similar_theta_stds)
    plt.fill_between(subset_question_nums, similar_theta_means - 3 * similar_theta_stds, similar_theta_means + 3 * similar_theta_stds, color='green', alpha=0.2)

    plt.xlabel('Number of Questions')
    plt.ylabel('Theta')
    plt.title('Theta Estimation vs. Number of Questions')
    plt.legend()
    plt.grid(True)
    plt.savefig('../plot/synthetic/random_test.png')