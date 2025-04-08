import numpy as np

def calculate_sensitivity_matrix(grid_size, electrode_positions, permittivity, perturbation=1e-3):
    """
    Calculates the sensitivity matrix for ECT.

    Parameters:
        grid_size (int): The size of the grid (e.g., 128 for a 128x128 grid).
        electrode_positions (list): List of electrode positions (x, y) in the domain.
        permittivity (numpy.ndarray): Initial permittivity distribution (2D array).
        perturbation (float): Small change in permittivity for sensitivity calculation.

    Returns:
        numpy.ndarray: Sensitivity matrix of shape (num_measurements, grid_size^2).
    """
    num_electrodes = len(electrode_positions)
    num_measurements = (num_electrodes * (num_electrodes - 1)) // 2
    sensitivity_matrix = np.zeros((num_measurements, grid_size**2))

    # Flatten the permittivity grid for easier indexing
    flat_permittivity = permittivity.flatten()

    # Loop through each grid point
    for idx in range(grid_size**2):
        # Perturb the permittivity at the current grid point
        perturbed_permittivity = flat_permittivity.copy()
        perturbed_permittivity[idx] += perturbation

        # Reshape back to 2D for simulation
        perturbed_permittivity_2d = perturbed_permittivity.reshape((grid_size, grid_size))

        # Simulate capacitance measurements with perturbed permittivity
        perturbed_measurements = simulate_capacitance(electrode_positions, perturbed_permittivity_2d)

        # Simulate capacitance measurements with original permittivity
        original_measurements = simulate_capacitance(electrode_positions, permittivity)

        # Calculate the sensitivity for this grid point
        sensitivity_matrix[:, idx] = (perturbed_measurements - original_measurements) / perturbation

    return sensitivity_matrix



def simulate_capacitance(electrode_positions, permittivity):
    """
    Simulates capacitance measurements for given electrode positions and permittivity.

    Parameters:
        electrode_positions (list): List of electrode positions (x, y) in the domain.
        permittivity (numpy.ndarray): Permittivity distribution (2D array).

    Returns:
        numpy.ndarray: Simulated capacitance measurements (1D array).
    """
    num_electrodes = len(electrode_positions)
    measurements = []

    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            # Compute the electric field path between electrodes i and j
            ex, ey = electrode_positions[i]
            fx, fy = electrode_positions[j]

            # Approximate the capacitance as the sum of permittivity along the line
            rr = np.linspace(ey, fy, 100).astype(int)
            cc = np.linspace(ex, fx, 100).astype(int)

            # Ensure indices are within bounds
            valid = (rr >= 0) & (rr < permittivity.shape[0]) & (cc >= 0) & (cc < permittivity.shape[1])
            rr, cc = rr[valid], cc[valid]

            # Sum permittivity along the valid path
            line_integral = permittivity[rr, cc].sum()
            measurements.append(line_integral)

    return np.array(measurements)