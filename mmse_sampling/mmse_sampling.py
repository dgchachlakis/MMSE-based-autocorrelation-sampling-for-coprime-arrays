import numpy as np
def mmse_sampling(He_matrix, P_matrix, Esel, number_of_snapshots):
    return np.linalg.inv(He_matrix + P_matrix / number_of_snapshots) @ He_matrix @ Esel
