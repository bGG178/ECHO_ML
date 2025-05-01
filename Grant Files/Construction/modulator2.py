import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

def generate_circulant_matrix_from_measurements(measurements, baseline, build_circulant_matrix, matrix_to_image, visualize=True):
    """
    Subtracts baseline from capacitance measurements, builds a circulant matrix,
    optionally visualizes it, and returns the matrix.

    Parameters:
    - measurements (list of float): Raw capacitance readings.
    - baseline (list of float): Baseline capacitance readings.
    - build_circulant_matrix (function): Function that converts 1D measurements into a circulant matrix.
    - matrix_to_image (function): Function that converts the matrix to a displayable image.
    - visualize (bool): If True, displays the matrix as an image.

    Returns:
    - np.ndarray: The generated circulant matrix.
    """

    # Normalize
    normalized = subtract_baseline(measurements,baseline)

    # Generate circulant matrix
    circ_mat = build_circulant_matrix(normalized)

    # Optional visualization
    if visualize:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Circulant Matrix Physical")
        ax.imshow(matrix_to_image(circ_mat), cmap='gray')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        plt.show()

    return circ_mat


def subtract_baseline(measurement_array, baseline_array):
    """
    Subtracts the baseline array (empty scan) from the measurement array.

    Args:
        measurement_array (array-like): Array from ECT scan with contents.
        baseline_array (array-like): Array from ECT scan with empty dewar (noise only).

    Returns:
        np.ndarray: Cleaned array with baseline noise subtracted.
    """
    # Convert to numpy arrays in case input isn't already
    measurement = np.array(measurement_array)
    baseline = np.array(baseline_array)

    # Sanity check: ensure arrays are same shape
    if measurement.shape != baseline.shape:
        raise ValueError("Measurement and baseline arrays must have the same shape")

    return measurement - baseline

def phantomplot(ax1, phantom):
    r = 5  # Match the electrode sensing area
    ax1.set_xlim(-r, r)
    ax1.set_ylim(-r, r)
    ax1.set_aspect('equal')
    ax1.set_title(f"Original Phantom")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    for obj in phantom:
        center = obj['center']
        radius = obj['radius']
        circle = plt.Circle(center, radius, edgecolor='blue', facecolor='none', linewidth=2)
        ax1.add_patch(circle)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Plot each object in the phantom
    for obj in phantom:
        center = obj['center']
        radius = obj['radius']
        circle = plt.Circle(center, radius, edgecolor='blue', facecolor='none', linewidth=2)
        ax1.add_patch(circle)

def matrix_to_image(C, output_path=None, normalize=True):
    """
    Converts a 2D numpy array C into a PIL grayscale image.
    If normalize=True, linearly scales C to [0,255].
    If output_path is provided, saves the image to that path.
    Returns the PIL Image object.
    """
    arr = C.astype(np.float32)
    if normalize:
        # Scale min→0, max→255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:  # Avoid division by zero
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)  # Set to zero if all values are the same
    # Convert to uint8
    img_arr = arr.astype(np.uint8)
    img = Image.fromarray(img_arr, mode='L')  # 'L' = (8‑bit pixels, black and white)
    if output_path:
        img.save(output_path)
    return img

def build_circulant_matrix(m):
    """
    Given a 1D array m of length N, returns the N×N circulant matrix
    whose k-th row is m right‑shifted by k positions.

    When you have a single 1xN measurement vector from your ECT sensor array,
    you can build rotation‑invariance simply by “virtually” rotating that vector through all
    possible cyclic shifts.
    """
    N = m.shape[0]
    C = np.zeros((N, N), dtype=m.dtype)
    for k in range(N):
        # Right cyclic shift by k: elements m[-k:], m[:-k]
        C[k] = np.roll(m, k)
    return C

if __name__ == "__main__":
    datafile = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData\npg_1_12e12_traintest.json"
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"File not found: {datafile}")

    with open(datafile, 'r') as f:
        combined_data = json.load(f)

    PLOTDATA = False
    physicalData = True

    if physicalData == False:

        # Convert all 'measurements' entries once before the loop
        for data in combined_data:
            data['measurements'] = np.array(data['measurements'])

        for i in range(len(combined_data)):
            raw_vec = combined_data[i]['measurements']
            phantom_data = combined_data[i]['objects']

            # 1) Build the circulant matrix
            circ_mat = build_circulant_matrix(raw_vec)



            if PLOTDATA:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                # Plot phantom and matrix
                phantomplot(ax1, phantom_data)
                ax2.set_title(f"Circulant Matrix {i}")
                ax2.imshow(matrix_to_image(circ_mat), cmap='gray')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')

                plt.tight_layout()
                plt.show()

    if physicalData == True:


        baseline = [0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0]
        measurements = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.108341, 0.307936, 0.0, 0.0, 0.0, 0.0, 0.0]
        npmeasurements = np.array(measurements)

        normalizedmeasurements = subtract_baseline(measurements, baseline)

        npnormalizedmeasurements = np.array(normalizedmeasurements)

        circ_mat = build_circulant_matrix(npnormalizedmeasurements)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot phantom and matrix
        ax2.set_title(f"Circulant Matrix Physical")
        ax2.imshow(matrix_to_image(circ_mat), cmap='gray')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.tight_layout()
        plt.show()

        # save circ_mat as image
        capacitance_path = r"C:\Users\welov\PycharmProjects\ECHO_ML\Grant Files\SavedModels\capacitance_image.npy"
        circ_mat_img = matrix_to_image(circ_mat)
        np.save(capacitance_path, circ_mat_img)