import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def phantomplot(phantom):
    r = 1  # Define the radius of the plotting area
    fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure and axis
    ax.set_xlim(-r, r)  # Set x-axis limits
    ax.set_ylim(-r, r)  # Set y-axis limits
    ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
    ax.set_title('Phantom Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot each object in the phantom
    for obj in phantom:
        center = obj['center']
        radius = obj['radius']
        circle = Circle(center, radius, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(circle)  # Add the circle to the plot

    plt.show()  # Display the plot


def get_measurement_count(electrodecount):
    """
    Given the number of electrodes, returns the number of measurements.
    The formula is based on the assumption that each electrode contributes
    to a unique measurement.
    """
    measurementcount = (electrodecount*(electrodecount - 1)) // 2
    #print(measurementcount)
    return measurementcount

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
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    # Convert to uint8
    img_arr = arr.astype(np.uint8)
    img = Image.fromarray(img_arr, mode='L')  # 'L' = (8‑bit pixels, black and white)
    if output_path:
        img.save(output_path)
    return img

if __name__ == "__main__":
    datafile = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData\combined_data.npy"
    loaded_data = np.load(datafile, allow_pickle=True)
    print("Loaded data example:", loaded_data[4]['measurements'])

    #--HYPERPARAMS--
    ELECTRODE_NUM = 15
    # Example raw measurement vector using random numbers

    for i in range(len(loaded_data)):
        raw_vec = loaded_data[i]['measurements']
        phantom_data = loaded_data[i]['objects']

        phantomplot(phantom_data)

        # 1) Build the circulant matrix
        circ_mat = build_circulant_matrix(raw_vec)

        # 2) Convert to image (and save to disk)
        img = matrix_to_image(circ_mat, output_path=f"circulant{get_measurement_count(ELECTRODE_NUM)}.png")

        # 3)  Plot display the image
        plt.title(f"Circulant Matrix {i}")
        plt.imshow(img)
        plt.show()
