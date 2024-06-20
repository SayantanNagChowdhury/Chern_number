# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:26:17 2024

@author: snagchowdh
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the lattice vectors
a1 = np.array([0, -1])
a2 = np.array([1, 0])
a3 = np.array([-1, 1])

# Define the matrix A(k)
def A(kx, ky):
    A_matrix = np.array([
        [0, -(1 + np.exp(1j * 2 * (kx * a2[0] + ky * a2[1]))), (1 + np.exp(-1j * 2 * (kx * a1[0] + ky * a1[1])))],
        [(1 + np.exp(-1j * 2 * (kx * a2[0] + ky * a2[1]))), 0, -(1 + np.exp(1j * 2 * (kx * a3[0] + ky * a3[1])))],
        [-(1 + np.exp(1j * 2 * (kx * a1[0] + ky * a1[1]))), (1 + np.exp(-1j * 2 * (kx * a3[0] + ky * a3[1]))), 0]
    ])
    return A_matrix

# Define the k-space grid
k_points = 100
kx_values = np.linspace(-np.pi, np.pi, k_points)
ky_values = np.linspace(-np.pi, np.pi, k_points)

# Compute the eigenvalues and eigenvectors for each point in k-space
eigenvalues = np.zeros((k_points, k_points, 3), dtype=complex)
eigenvectors = np.zeros((k_points, k_points, 3, 3), dtype=complex)

for i, kx in enumerate(kx_values):
    for j, ky in enumerate(ky_values):
        eigvals, eigvecs = np.linalg.eig(A(kx, ky))
        eigenvalues[i, j, :] = eigvals
        eigenvectors[i, j, :, :] = eigvecs

# Function to calculate Berry connection and Berry curvature
def berry_connection(eigenvectors, i, j, band, mu):
    dk = kx_values[1] - kx_values[0]  # Assume uniform grid spacing for dkx and dky
    if mu == 0:  # Partial derivative with respect to kx
        psi_plus = eigenvectors[i+1, j, :, band]
        psi = eigenvectors[i, j, :, band]
    elif mu == 1:  # Partial derivative with respect to ky
        psi_plus = eigenvectors[i, j+1, :, band]
        psi = eigenvectors[i, j, :, band]
    else:
        raise ValueError("mu must be 0 (kx) or 1 (ky)")
    
    return np.vdot(psi, psi_plus - psi) / dk

def berry_curvature(eigenvectors, i, j, band):
    A_kx = berry_connection(eigenvectors, i, j, band, 0)
    A_ky = berry_connection(eigenvectors, i, j, band, 1)
    A_kx_prev = berry_connection(eigenvectors, i-1, j, band, 0)
    A_ky_prev = berry_connection(eigenvectors, i, j-1, band, 1)
    
    dA_kx_dky = (A_kx - A_kx_prev) / (2 * (ky_values[1] - ky_values[0]))
    dA_ky_dkx = (A_ky - A_ky_prev) / (2 * (kx_values[1] - kx_values[0]))
    
    F12 = dA_ky_dkx - dA_kx_dky
    return np.imag(F12)

# Calculate the Chern numbers using Berry curvature
chern_numbers = []

# Loop over each band
for band in range(3):
    curvature = np.zeros((k_points, k_points), dtype=complex)
    
    # Compute Berry curvature for the band
    for i in range(1, k_points - 1):
        for j in range(1, k_points - 1):
            curvature[i, j] = berry_curvature(eigenvectors, i, j, band)
    
    # Compute Chern number using Equation 7a
    chern_number = 0.0
    for i in range(1, k_points - 1):
        for j in range(1, k_points - 1):
            chern_number += curvature[i, j]
    
    # Normalize by 2*pi to get the Chern number
    chern_number *= (kx_values[1] - kx_values[0]) * (ky_values[1] - ky_values[0]) / (2 * np.pi)
    
    # Round and store the Chern number
    chern_numbers.append(np.round(chern_number.real))

print("Chern numbers for each band (rounded):", chern_numbers)

# Create a meshgrid for kx and ky values
kx_grid, ky_grid = np.meshgrid(kx_values, ky_values)

# Flatten the kx, ky grid for 3D plotting
kx_flat = kx_grid.flatten()
ky_flat = ky_grid.flatten()

# Plot the band structure with Chern numbers in the legend
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']

for band in range(3):
    eigenvalues_band_flat = eigenvalues[:, :, band].flatten()
    ax.scatter(kx_flat / np.pi, ky_flat / np.pi, eigenvalues_band_flat.imag, color=colors[band], label=f'Band {band+1} (Chern: {chern_numbers[band]})')

ax.set_xlabel('$k_x / \pi$')
ax.set_ylabel('$k_y / \pi$')
ax.set_zlabel('Imaginary part of eigenvalues')
ax.set_title('Band Structure with Chern Numbers')
ax.legend()
plt.show()
