import torch

# overleaf/R_library: z1/g, z2/a1, z3/d
def item_response_fn_3PL(z1, z2, z3, theta):
    return z1 + (1 - z1) / (1 + torch.exp(-(z2 * theta + z3)))

def item_response_fn_2PL(z2, z3, theta):
    return 1 / (1 + torch.exp(-(z2 * theta + z3)))

def item_response_fn_1PL(z3, theta):
    return 1 / (1 + torch.exp(-(theta + z3)))

