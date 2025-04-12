import numpy as np
import matplotlib.pyplot as plt
from sens_matrix import calculate_sensitivity_matrix
from phantom_generator import PhantomGenerator

def linear_back_projection(measurements, sensitivity_matrix): # back project the measurement data for image reconstruction
    """
    Reconstructs an image using Linear Back Projection (LBP).

    Parameters:
        measurements (numpy.ndarray): The capacitance measurements (1D array).
        sensitivity_matrix (numpy.ndarray): The sensitivity matrix (2D array).

    Returns:
        numpy.ndarray: The reconstructed image.
    """
    # there are a few problems with this statement and how linear back projection actually works.
    # 1. you should be generating the sensitivity matrix to determine how much to weight the data form each electrode
    # 2. the capacitance measurements should not be a "1D array", they should also be 2D

    # Perform the linear back projection
    reconstructed_image = np.dot(sensitivity_matrix.T, measurements) # you need to do element wise multiplication
    # between the sensitivity matrix and the measurement matrix. they should be the same size

    # Debug: Check the raw reconstructed image
    print("Raw reconstructed image stats:", reconstructed_image.min(), reconstructed_image.max())

    # Normalize the reconstructed image to [0, 1]
    reconstructed_image -= reconstructed_image.min() # this should be partially done by the sensitivity matrix
    reconstructed_image /= reconstructed_image.max()

    return reconstructed_image

if __name__ == "__main__":
    # Load the modulated data (measurements and phantom objects)
    datafile = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData\combined_data.npy"
    loaded_data = np.load(datafile, allow_pickle=True)

    # Define parameters
    grid_size = 128 # image size (square image)
    electrode_count = 12
    domain_radius = 1.0 # i bet r in phantom_generator.py is the domain radius

    # Generate the sensitivity matrix
    permittivity = np.ones((grid_size, grid_size))  # Uniform permittivity
    electrode_positions = PhantomGenerator(electrode_count, domain_radius, grid_size).electrodes # collect electrode positions
    sensitivity_matrix = calculate_sensitivity_matrix(grid_size, electrode_positions, permittivity) # calculate the sensitivity matrix form what?

    # Debug: Check sensitivity matrix
    print("Sensitivity matrix stats:", sensitivity_matrix.min(), sensitivity_matrix.max()) # how is the sensitivity matrix being checked?
    plt.imshow(sensitivity_matrix, aspect='auto', cmap='viridis')
    plt.title("Sensitivity Matrix")
    plt.colorbar()
    plt.show()

    # Loop through each phantom and reconstruct it
    for i, data in enumerate(loaded_data):
        measurements = data['measurements']  # Modulated capacitance measurements
        objects = data['objects']  # Phantom objects

        # Debug: Check measurements
        print(f"Measurements for Phantom {i} stats:", measurements.min(), measurements.max())

        # Reconstruct the original phantom from the stored objects
        original_phantom = PhantomGenerator(electrode_count, domain_radius, grid_size).generate_phantom(objects)

        # Perform Linear Back Projection
        reconstructed_image = linear_back_projection(measurements, sensitivity_matrix)

        # Reshape the reconstructed image for visualization
        reconstructed_image = reconstructed_image.reshape((grid_size, grid_size))

        # Plot the original phantom and reconstructed image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(original_phantom, cmap='gray', extent=[-domain_radius, domain_radius, -domain_radius, domain_radius])
        ax1.set_title(f"Original Phantom {i}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        ax2.imshow(reconstructed_image, cmap='gray', extent=[-domain_radius, domain_radius, -domain_radius, domain_radius])
        ax2.set_title(f"Reconstructed Phantom {i}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

        plt.tight_layout()
        plt.show()