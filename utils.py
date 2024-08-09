import torch

def item_response_fn_3PL(z1, z2, z3, theta):
    p_correct = z1 + (1 - z1) / (1 + torch.exp(-(z2 * theta + z3)))
    return p_correct