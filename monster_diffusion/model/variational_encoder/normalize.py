def normalize(tensor):
    flat_tensor = tensor.flatten(start_dim=1)
    return (
        flat_tensor.sub(flat_tensor.mean(dim=1, keepdims=True))
        .div(flat_tensor.std(dim=1, keepdims=True).add(1e-8))
        .view_as(tensor)
    )
