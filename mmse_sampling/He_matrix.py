import numpy as np
def He_matrix(source_powers, noise_power, p, integrals_map, channel):
    (carrier_frequency, propagation_speed) = channel
    wavelength = propagation_speed / carrier_frequency
    unit_spacing = wavelength / 2
    p = p / unit_spacing
    c = np.linalg.norm(source_powers[:, None], 'fro') ** 2
    array_length = p.shape[0]
    pdot = (np.kron(np.ones(array_length, ), p)).astype(int)
    pddot = (np.kron(p, np.ones(array_length, ))).astype(int)
    omega = (pdot - pddot).astype(int)
    T = array_length ** 2
    He = np.zeros((T, T))
    for i in range(T):
        for m in range(T):
            He[i, m] = (c * integrals_map[omega[i] - omega[m]] +
                       noise_power ** 2 * delta(omega[i]) * delta(omega[m]) +
                       noise_power * np.sum(source_powers) * (delta(omega[i]) * integrals_map[-omega[m]] + delta(omega[m]) * integrals_map[omega[i]]) + 
                       integrals_map[omega[i]] * integrals_map[-omega[m]] * (np.sum(source_powers) ** 2 - c))      
    return He
def delta(x):
    return x == 0
