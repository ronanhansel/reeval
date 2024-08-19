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
from utils import item_response_fn_1PL
import numpy as np

# export JAX_PLATFORM_NAME=cpu
# nohup python CAT.py > /scratch/yuhengtu/workspace/certified-eval/src/CAT.log 2>&1 &

def clear_caches():
    modules = list(sys.modules.items())  # Create a list of items to avoid runtime errors
    for module_name, module in modules:
        if module_name.startswith("jax"):
            if module_name not in ["jax.interpreters.partial_eval"]:
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        try:
                            obj.cache_clear()
                        except:
                            pass
    gc.collect()
    print("cache cleared")
  
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
        
        # bernoulli = torch.distributions.Bernoulli(prob)
        # sample = bernoulli.sample((1,))
        # log_prob = bernoulli.log_prob(sample)
        # grad = torch.autograd.grad(log_prob, theta, create_graph=True, retain_graph=True)[0]
        # hessian = torch.autograd.grad(grad, theta, retain_graph=True)[0]
        # fisher_info_list.append((-1)*hessian)
        
        # samples_Y = bernoulli.sample((100,))
        # hessian_list = [] # doesn't depend on sample_y
        # for sample_y in samples_Y:
        #     if theta.grad is not None:
        #         theta.grad.zero_() 
        #     log_prob = bernoulli.log_prob(sample_y)
        #     grad = torch.autograd.grad(log_prob, theta, create_graph=True, retain_graph=True)[0]
        #     hessian = torch.autograd.grad(grad, theta, retain_graph=True)[0]
        #     hessian_list.append(hessian)
        # fisher_info = (-1) * torch.tensor(hessian_list).mean()
        # fisher_info_list.append(fisher_info)

    unasked_question_index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
    return unasked_question_list[unasked_question_index_with_max_fisher_info]

def CAT_modern(z3, unasked_question_list, theta_samples):
    theta_samples_tensor = torch.tensor(np.array(theta_samples))
    fisher_info_list = []
    for unasked_question_index in unasked_question_list:
        # fisher_info_samples = []
        # for theta_sample in theta_samples:
        #     theta = torch.tensor(theta_sample.item(), requires_grad=True)
        #     z_single = z3[unasked_question_index].clone().detach()
        #     prob = item_response_fn_1PL(z_single, theta)
        #     # bernoulli = torch.distributions.Bernoulli(prob)
        #     # sample = bernoulli.sample((1,))
        #     # log_prob = bernoulli.log_prob(sample)
        #     # grad = torch.autograd.grad(log_prob, theta, create_graph=True, retain_graph=True)[0]
        #     # hessian = torch.autograd.grad(grad, theta, retain_graph=True)[0]
        #     hessian = prob * (1 - prob)
        #     fisher_info_samples.append((-1)*hessian)
        # fisher_info_list.append(torch.tensor(fisher_info_samples).mean())
        
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
    print(f'init_question_index: {init_question_index}')
    init_answer = new_testtaker.ask(z3, init_question_index)

    unasked_question_list = [i for i in range(question_num)]
    unasked_question_list.remove(init_question_index)
    asked_answer_list = [init_answer.float()]
    asked_z3_list = [z3[init_question_index]]

    theta_means = []
    theta_stds = []
    for epoch in range(subset_question_num):
        print(f'\nepoch: {epoch+1}')
        asked_answer_jnp = jnp.array(asked_answer_list)
        asked_z3_jnp = jnp.array(asked_z3_list)

        mean_theta, std_theta, theta_samples = fit_theta_mcmc(
            asked_z3_jnp, 
            asked_answer_jnp
            )
        theta_means.append(mean_theta.item())
        print(f"theta mean: {theta_means}")
        theta_stds.append(std_theta.item())

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
        
        print(f'new_question_index: {new_question_index}')
        new_answer = new_testtaker.ask(z3, new_question_index)
        unasked_question_list.remove(new_question_index)
        asked_answer_list.append(new_answer.float())
        asked_z3_list.append(z3[new_question_index])
        
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
    print(f'z3: {z3}')
    
    print(f'num of total questions: {question_num}')
    
    new_testtaker = SimulatedTestTaker(theta=1.25, model="1PL")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")

    subset_question_num = 100
    random_theta_means, random_theta_stds = main(z3, new_testtaker, strategy="random", subset_question_num=subset_question_num)
    owen_theta_means, owen_theta_stds = main(z3, new_testtaker, strategy="owen", subset_question_num=subset_question_num)
    fisher_theta_means, fisher_theta_stds = main(z3, new_testtaker, strategy="fisher", subset_question_num=subset_question_num)
    modern_theta_means, modern_theta_stds = main(z3, new_testtaker, strategy="modern", subset_question_num=subset_question_num)
    
    total_question_nums = range(question_num)
    subset_question_nums = range(subset_question_num)
    
    plt.figure(figsize=(10, 6))
    plt.plot(subset_question_nums, [theta_star] * subset_question_num, label='True Theta', color='black', linestyle='--')
    
    plt.plot(subset_question_nums, random_theta_means, label='Random Testing', color='blue')
    random_theta_means = jnp.array(random_theta_means)
    random_theta_stds = jnp.array(random_theta_stds)
    plt.fill_between(
        subset_question_nums, 
        random_theta_means - 3 * random_theta_stds, 
        random_theta_means + 3 * random_theta_stds, 
        color='blue', 
        alpha=0.2
    )
    
    plt.plot(subset_question_nums, owen_theta_means, label='CAT with owen', color='green')
    owen_theta_means = jnp.array(owen_theta_means)
    owen_theta_stds = jnp.array(owen_theta_stds)
    plt.fill_between(subset_question_nums, 
                     owen_theta_means - 3 * owen_theta_stds, 
                     owen_theta_means + 3 * owen_theta_stds, 
                     color='green', 
                     alpha=0.2
                     )

    plt.plot(subset_question_nums, fisher_theta_means, label='CAT with fisher', color='red')
    fisher_theta_means = jnp.array(fisher_theta_means)
    fisher_theta_stds = jnp.array(fisher_theta_stds)
    plt.fill_between(subset_question_nums, 
                     fisher_theta_means - 3 * fisher_theta_stds, 
                     fisher_theta_means + 3 * fisher_theta_stds, 
                     color='red', 
                     alpha=0.2
                     )

    plt.plot(subset_question_nums, modern_theta_means, label='CAT with modern', color='purple')
    modern_theta_means = jnp.array(modern_theta_means)
    modern_theta_stds = jnp.array(modern_theta_stds)
    plt.fill_between(subset_question_nums, 
                     modern_theta_means - 3 * modern_theta_stds, 
                     modern_theta_means + 3 * modern_theta_stds, 
                     color='purple', 
                     alpha=0.2
                     )
    
    plt.xlabel('Number of Questions')
    plt.ylabel('Theta')
    plt.title('Theta Estimation vs. Number of Questions')
    plt.legend()
    plt.grid(True)
    plt.savefig('../plot/synthetic/random_adaptive_test.png')
    
    print("finish!!!")