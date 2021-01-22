import numpy as np
def generate_uniform_doas(K):
    th = np.pi  * np.random.rand(K, ) - np.pi / 2
    return np.sort(th)