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


def subtract_baseline(measurement_array, baseline_array,epsilon=1e-15):
    """
    Divides the measurement array by the baseline array element-wise.

    Args:
        measurement_array (array-like): Array from ECT scan with contents.
        baseline_array (array-like): Array from ECT scan with empty dewar (noise only).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Array with each element as measurement / baseline.
    """
    measurement = np.array(measurement_array)
    baseline = np.array(baseline_array)

    if measurement.shape != baseline.shape:
        raise ValueError("Measurement and baseline arrays must have the same shape")

    return measurement / (baseline + epsilon)  # epsilon avoids division by zero

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

        cyl1 = [0.9748556000947695, 1.0158282647606631, 0.8330202658169318, 0.6465209219977359, 0.12256027572784957, 0.08890590189069338, 0.23260472578221494, 0.035815572479549314, 0.4695769691582139, 0.8116955108130423, 0.2795957899545081, 0.2456511500871949, 0.0, 0.3078962677200969, 0.9613355573762375, 0.8705941631669781, 0.5853834315864682, 0.24897156230700898, 0.07092631749273563, 0.023669845987407657, 0.2933893083729245, 0.7893394096365414, 0.3908308081744082, 0.3216977031071976, 0.12403928896873873, 0.030458185701616354, 0.0, 0.4981453989414405, 0.4830030286950152, 0.4389819106393869, 0.30987169840339196, 0.06993428519548579, 0.15249154227515416, 0.020328453081221096, 0.25202292085812905, 0.7204119536801427, 0.6363636291270897, 0.35297694921700523, 0.0, 0.985924935278608, 0.9384588688410865, 0.7792835395016817, 0.48599436821814457, 0.0, 0.17752383980299244, 0.057142855803716856, 0.8351720471873292, 0.37132973527936297, 1.0990051225185165, 0.02547910524999533, 0.9943752241069914, 0.9763531500749864, 0.8992315132659635, 0.4340584798247285, 0.013245787501649066, 0.0598277717575677, 2.000420215646156, 0.8786598701926882, 0.5103270313320472, 0.14037208628359735, 0.967829962379796, 0.8868828171190689, 0.31542194264610857, 0.18182437978202953, 0.08889456095024484, 0.1472470476250568, 0.037249207003714625, 0.8125661557683489, 0.0, 0.9942472088361838, 0.9102178892516595, 0.7660307352649971, 0.5226356984178576, 0.015504911012590751, 0.6850257159852431, 0.24159491447228434, 0.0, 0.9519588069559728, 0.8657174786358521, 0.7235303344629844, 0.37011812059208965, 0.05369437793405656, 0.07583248097528111, 0.0, 0.991454271935415, 0.9617986856279429, 0.8851110721905034, 0.7172018257944403, 0.626780186813138, 0.0, 0.9917016498160064, 0.9545165238613201, 0.8466433528539071, 0.6519906545298363, 0.0, 0.0, 0.0, 0.23653064130935703, 0.0, 0.981518970308278, 0.9234388321001451, 0.023739480424481665, 0.9932248158266643, 0.0, 0.0]

        normalizedmeasurements = subtract_baseline(measurements, baseline)

        npnormalizedmeasurements = np.array(cyl1)

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