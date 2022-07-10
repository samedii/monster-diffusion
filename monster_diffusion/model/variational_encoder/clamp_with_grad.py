def clamp_with_grad(tensor, min=-3.7, max=3.7):
    return (tensor - tensor.detach()) + tensor.detach().clamp(min, max)
