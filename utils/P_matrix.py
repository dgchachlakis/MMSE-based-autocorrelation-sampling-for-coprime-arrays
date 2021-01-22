import numpy as np
def P_matrix(source_powers, noise_power, p, integrals_map, channel):
    (carrier_frequency, propagation_speed) = channel
    wavelength = propagation_speed / carrier_frequency
    unit_spacing = wavelength / 2
    c = np.linalg.norm(source_powers[:, None], 'fro') ** 2
    array_length = p.shape[0]
    T = array_length ** 2
    p = p / unit_spacing
    number_of_sources = source_powers.shape[0]
    pdot = (np.kron(np.ones(array_length, ), p )).astype(int)
    pddot = (np.kron(p, np.ones(array_length, ))).astype(int)
    s = np.array(range(1, number_of_sources + array_length + 1))
    udot = (np.kron(np.ones(number_of_sources + array_length, ), s)).astype(int)
    uddot = (np.kron(s, np.ones(number_of_sources + array_length, ))).astype(int)
    s = np.array(range(1, array_length + 1))
    vdot = (np.kron(np.ones(array_length, ), s)).astype(int)
    vddot = (np.kron(s, np.ones(array_length, ))).astype(int)
    Omd = (np.outer(pdot, np.ones((1, T))) - np.outer(np.ones((1, T)), pdot)).astype(int)
    Omdd = (np.outer(pddot, np.ones((1, T))) - np.outer(np.ones((1, T)), pddot)).astype(int)
    P = np.zeros((T, T))
    source_powers=np.concatenate((np.array([0]),source_powers), axis = 0)
    for i in range(T):
        for m in range(T):
            for j in range((number_of_sources + array_length) ** 2):
                if udot[j] == uddot[j] <= number_of_sources:
                    P[i, m] += source_powers[uddot[j]] * source_powers[udot[j]] * integrals_map[Omdd[m, i] + Omd[i, m]] 
                elif udot[j] <= number_of_sources and uddot[j] <= number_of_sources and udot[j] != uddot[j]:
                    P[i, m] += source_powers[uddot[j]] * source_powers[udot[j]] * integrals_map[Omdd[m, i]] * integrals_map[Omd[i, m]]
                elif udot[j] <= number_of_sources and uddot[j] - number_of_sources == vddot[i] == vddot[m]:
                    P[i, m] += noise_power * source_powers[udot[j]] * integrals_map[Omdd[m, i]]
                elif uddot[j] <= number_of_sources and udot[j] - number_of_sources == vdot[i] == vdot[m]:
                    P[i, m] += noise_power * source_powers[uddot[j]] * integrals_map[Omd[i, m]]
                elif udot[j] - number_of_sources == vdot[i] == vdot[m] and uddot[j] - number_of_sources == vddot[i] == vddot[m]:
                    P[i, m] += noise_power ** 2
                else:
                    P[i, m] += 0
    return P