import numpy as np
import scipy.special as sp
def form_integrals_map(M, N, channel):
    (carrier_frequency, propagation_speed) = channel
    wavelength = propagation_speed / carrier_frequency
    unit_spacing = wavelength / 2
    int_map={}
    for n in range(-4 * M * N + 2 * N, 4 * M * N - 2 * N + 1):
        int_map[n] = sp.jv(0, unit_spacing * n * 2 * np.pi * carrier_frequency / propagation_speed) 
    return int_map