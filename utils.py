import torch

def item_response_fn_3PL(a, b, c, theta):
    """
    Item response function: 
    """
    exp_part = torch.exp(a * (theta - b))
    p_correct = c + (1 - c) * (exp_part / (1 + exp_part))
    return p_correct