import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
random.seed(42)
torch.manual_seed(42)

def item_response_fn_1PL(d, theta):
    return 1 / (1 + torch.exp(-(theta + d)))

def get_new_test_taker_answer(question_index):
    new_test_taker = pd.read_csv('./clean_data/divided_base/divided_base_01-ai_yi-34b-chat_0.csv')
    answer = new_test_taker.iloc[question_index, -1]
    answer = torch.tensor(answer)
    return answer

def fit_theta(Z, asked_question_list, asked_answer_list, epoch=100):
    theta_hat = torch.rand(1, requires_grad=True)
    optimizer = optim.Adam([theta_hat], lr=0.01)
    for _ in range(epoch):
        log_prob = 0
        for i, asked_question_index in enumerate(asked_question_list):
            prob = item_response_fn_1PL(Z[asked_question_index], theta_hat)
            bernoulli = torch.distributions.Bernoulli(prob)
            log_prob = log_prob + bernoulli.log_prob(asked_answer_list[i])
        log_prob.backward()
        optimizer.step()
        optimizer.zero_grad()
    return theta_hat

# def main(Y):
df = pd.read_csv('./model_coef/divided_base_coef_1PL_clean.csv', usecols=[1])
Z = df.iloc[:, 0].tolist()

I = len(Z)
print(f'num of total questions: {I}')

K = 10

init_question_index = random.randint(0, I - 1)
init_answer = get_new_test_taker_answer(init_question_index)

unasked_question_list = [i for i in range(I)]
unasked_question_list.remove(init_question_index)
asked_question_list = [init_question_index]
asked_answer_list = [init_answer]

for k in range(K):
    theta_hat = fit_theta(Z, asked_question_list, asked_answer_list)

    fisher_info_list = []

    for unasked_question_index in unasked_question_list:
        prob = item_response_fn_1PL(Z[unasked_question_index], theta_hat)
        bernoulli = torch.distributions.Bernoulli(prob)
        samples_Y = bernoulli.sample((100,))
        hessian_list = [] # all the same

        for sample in samples_Y:
            prob = item_response_fn_1PL(Z[unasked_question_index], theta_hat)
            bernoulli = torch.distributions.Bernoulli(prob)
            log_prob = bernoulli.log_prob(sample)

            grad = torch.autograd.grad(log_prob, theta_hat, create_graph=True)[0]
            hessian = torch.autograd.grad(grad, theta_hat)[0]
            hessian_list.append(hessian)

        fisher_info = torch.stack(hessian_list).mean()
        fisher_info_list.append(fisher_info)

    unasked_question_index_with_max_fisher_info = torch.argmax(torch.tensor(fisher_info_list)).item()
    new_question_index = unasked_question_list[unasked_question_index_with_max_fisher_info]
    print(new_question_index)

    new_answer = get_new_test_taker_answer(new_question_index)
    unasked_question_list.remove(new_answer)
    asked_question_list.append(new_question_index)
    asked_answer_list.append(new_answer)

    # new_test_taker.ask(question_with_max_fisher_info)
    # unasked_questions.remove(question_with_max_fisher_info)

# return theta_hat

