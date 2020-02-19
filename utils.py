def one_hot_vector(length, indices):
    vec = [0 for i in range(0, length)]
    if not isinstance(indices, list):
        indices = [indices]
    for i in indices:
        vec[i] = 1
    return vec
