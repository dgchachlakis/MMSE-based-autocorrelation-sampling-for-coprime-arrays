import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mmse_sampling import *
# channel
carrier_frequency = 1.5 * 10 ** 8
propagation_speed = 3 * 10 ** 8
wavelength = propagation_speed / carrier_frequency
unit_spacing = wavelength / 2
channel = (carrier_frequency, propagation_speed)
# Coprime array with coprimes M, N such that M < N
M = 2
N = 3
p = ca_element_locations(M, N, channel) 
# source and noise powers
source_powers = np.array([10, 5, 10, 5])
number_of_sources = source_powers.shape[0]
noise_power = 1
# autocorrelation sampling matrices
Jdict = form_index_sets(M, N , pair_wise_distances(p), channel)
Esel = selection_sampling(Jdict, array_length(M, N), coarray_length(M, N))
Eavg = averaging_sampling(Jdict, array_length(M, N), coarray_length(M, N))
# Matrices required for forming Emmse
int_map = form_integrals_map(M, N, channel)
He = He_matrix(source_powers, noise_power, p, int_map, channel)
P = P_matrix(source_powers, noise_power, p, int_map, channel)
# Smoothing matrix
F = smoothing_matrix(coarray_length(M, N))
# Sample support axis and number of realizations
number_of_snapshots_axis = [10, 20, 40, 60, 80, 100]
number_of_realizations = 1500
# Zero - padding
err_sel = np.zeros((len(number_of_snapshots_axis), number_of_realizations))
err_avg = np.zeros((len(number_of_snapshots_axis), number_of_realizations))
err_mmse = np.zeros((len(number_of_snapshots_axis), number_of_realizations))
for j in range(number_of_realizations):
    # DoA sources
    thetas = generate_uniform_doas(number_of_sources)
    # Array response matrix
    S = response_matrix(thetas, p, channel)
    # Nominal Physical autocrrelation matrix
    R = autocorrelation_matrix(S, source_powers, noise_power)
    # Nominal coarray autocorrelation matrix
    Z = spatial_smoothing(F, Esel.T @ R.flatten())
    for i, Q in enumerate(number_of_snapshots_axis):
        Emmse = mmse_sampling(He, P, Esel, Q)
        Y = snapshots(S, source_powers, noise_power, Q)
        Rest = autocorrelation_matrix_est(Y)
        r = Rest.flatten()
        Zsel = spatial_smoothing(F, Esel.T @ r)
        Zavg = spatial_smoothing(F, Eavg.T @ r)
        Zmmse = spatial_smoothing(F, Emmse.T @ r)
        err_sel[i , j] = np.linalg.norm(Z - Zsel, 'fro') ** 2
        err_avg[i , j] = np.linalg.norm(Z - Zavg, 'fro') ** 2
        err_mmse[i , j] = np.linalg.norm(Z - Zmmse, 'fro') ** 2
# Compute the sample-average MSE of each method
err_sel = np.mean(err_sel, axis = 1)
err_avg = np.mean(err_avg, axis = 1)
err_mmse = np.mean(err_mmse, axis = 1)
# Plot and compare MSEs 
plt.figure()
plt.plot(number_of_snapshots_axis, err_sel, '+-r', label = "Selection")
plt.plot(number_of_snapshots_axis, err_avg, '^-b', label = "Averaging")
plt.plot(number_of_snapshots_axis, err_mmse, 'x-k', label = "MMSE (proposed)")
plt.legend()
plt.grid(color='k', linestyle=':', linewidth=1)
plt.ylabel('MSE')
plt.xlabel('Sample support')
plt.title('Random DoAs following uniform distribution')
plt.show()