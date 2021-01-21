import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mse_expressions import *
# channel
carrier_frequency = 1 * 10 ** 7
propagation_speed = 3 * 10 ** 8
channel = (carrier_frequency, propagation_speed)
# Coprime array with coprimes M, N such that M < N
M = 2
N = 3
p = ca_element_locations(M, N, channel) 
# DoA sources
thetas = np.array([-np.pi / 3, -np.pi / 4, np.pi / 5, 2 * np.pi / 5])
# source and noise powers
source_powers = np.array([10, 10, 10, 10])
noise_power = 1
# Array response matrix
S = response_matrix(thetas, p, channel)
# Nominal Physical autocrrelation matrix
R = autocorrelation_matrix(S, source_powers, noise_power)
# autocorrelation sampling matrices
Jdict = form_index_sets(M, N , pair_wise_distances(p), channel)
Esel = selection_sampling(Jdict, array_length(M, N), coarray_length(M, N))
Eavg = averaging_sampling(Jdict, array_length(M, N), coarray_length(M, N))
# Nominal coarray autocorrelation matrix
F = smoothing_matrix(coarray_length(M, N))
Z = spatial_smoothing(F, Esel.T @ R.flatten())
# Sample support axis and number of realizations
number_of_snapshots_axis = [1, 100, 200, 300, 400]
number_of_realizations = 1500
# Zero - padding
err_sel_theory = np.zeros(len(number_of_snapshots_axis), )
err_avg_theory = np.zeros(len(number_of_snapshots_axis), )
err_sel_numerical = np.zeros((len(number_of_snapshots_axis), number_of_realizations))
err_avg_numerical = np.zeros((len(number_of_snapshots_axis), number_of_realizations))
for i, Q in enumerate(number_of_snapshots_axis):
    # Evaluate the theoretical MSE expressions only once per value of sample support
    err_sel_theory[i] = selection_mse(source_powers, noise_power, coarray_length(M, N), Q)
    err_avg_theory[i] = averaging_mse(thetas, source_powers, noise_power, coarray_length(M, N), Q, Jdict, p, channel)
    # For each value of sample support, compute empirically the MSE
    for j in range(number_of_realizations): 
        Y = snapshots(S, source_powers, noise_power, Q)
        Rest = autocorrelation_matrix_est(Y)
        r = Rest.flatten()
        Zsel = spatial_smoothing(F, Esel.T @ r)
        Zavg = spatial_smoothing(F, Eavg.T @ r)
        err_sel_numerical[i , j] = np.linalg.norm(Z-Zsel, 'fro') ** 2
        err_avg_numerical[i , j] = np.linalg.norm(Z-Zavg, 'fro') ** 2
# Compute the sample-average MSE
err_sel_numerical = np.mean(err_sel_numerical, axis = 1)
err_avg_numerical = np.mean(err_avg_numerical, axis = 1)
# Plot and compare MSE (theory) with estimated MSE (numerical)
plt.figure()
plt.semilogy(number_of_snapshots_axis, err_sel_theory, 'o-r', markerfacecolor = 'w', label = "Selection (Theory)")
plt.semilogy(number_of_snapshots_axis, err_sel_numerical, '+-r', label = "Selection (Numerical)")
plt.semilogy(number_of_snapshots_axis, err_avg_theory, 's-k', markerfacecolor = 'w', label = "Averaging (Theory)")
plt.semilogy(number_of_snapshots_axis, err_avg_numerical, 'x-k', label = "Averaging (Numerical)")
plt.legend()
plt.ylabel('MSE')
plt.xlabel('Sample support')
plt.show()