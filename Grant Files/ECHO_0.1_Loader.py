import random

import torch
from CGAN_paper_ECHO_1 import Generator
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
from Construction.modulator2 import generate_circulant_matrix_from_measurements, subtract_baseline, build_circulant_matrix, matrix_to_image

def load_generator_model(model_path="generator_model.pth", image_size=64, device=None):
    """
    Loads a trained generator model from file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator(image_size=image_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_image_from_capacitance(model, capacitance_array, image_size=64, device=None):
    """
    Generates a phantom image given a capacitance matrix input.

    Parameters:
    - model: Trained generator model
    - capacitance_array: 2D numpy array (circulant capacitance matrix)
    - image_size: Resolution for image generation
    - device: 'cuda' or 'cpu'

    Returns:
    - generated_img: 2D numpy array (generated image)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Preprocess the input
    cap_tensor = torch.tensor(capacitance_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cap_tensor = (cap_tensor - 0.5) * 2  # Normalize to [-1, 1]
    cap_tensor = F.interpolate(cap_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)
    cap_tensor = cap_tensor.to(device)

    # Generate the image
    with torch.no_grad():
        generated = model(cap_tensor)

    # Post-process: convert from [-1, 1] to [0, 1] and squeeze
    generated = generated.squeeze().cpu().numpy()
    generated = (generated + 1) / 2  # Rescale to [0, 1]
    return generated

Z = 0.1  # seconds between updates

def get_live_measurements():
    """
    Replace this function with real sensor input or stream reader.
    Here we mock it with noise for demonstration.
    """


    base_measurements = [0.0, 0.2, 0.122, 0.1, 0.0, 0.0, 0.0, 0.108341, 0.307936, 0.0, 0.0, 0.0, 0.0, 0.0]
    noise = np.random.normal(0, 0.01, size=len(base_measurements))
    measurements = np.array(base_measurements) + noise
    return measurements.tolist()


def generate_and_plot_realtime(model, baseline, image_size=64, interval=2, capsize=14, device=None):
    """
    Continuously generate images from live measurements and update plot.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    im_phantom = ax1.imshow(np.zeros((image_size, image_size)), cmap='gnuplot', vmin=0, vmax=1)
    ax1.set_title("Generated Phantom (Live)")
    ax1.axis("off")

    im_matrix = ax2.imshow(np.zeros((image_size, image_size)), cmap='viridis', vmin=0, vmax=1)
    ax2.set_title("Circulant Matrix")
    ax2.axis("off")

    plt.show()
    try:
        while True:
            # Step 1: Get new measurements
            measurements = get_live_measurements()

            # Step 2: Normalize and build matrix
            normalized = subtract_baseline(measurements, baseline)
            circ_mat = build_circulant_matrix(np.array(normalized))

            # Step 3: Run through model
            output = generate_image_from_capacitance(model, circ_mat, image_size=image_size, device=device)

            # Step 4: Resize matrix before conversion to image
            mat_tensor = torch.tensor(circ_mat).unsqueeze(0).unsqueeze(0).float()
            # For display — nearest preserves diagonal structure
            mat_resized = F.interpolate(mat_tensor, size=(capsize, capsize), mode='nearest')
            resized_mat = mat_resized.squeeze().numpy()

            # Check for flat matrix
            if np.allclose(resized_mat.min(), resized_mat.max()):
                print("⚠️ Warning: resized_mat has no dynamic range — will display as uniform color.")

            # Convert to image and back to NumPy for display
            img = matrix_to_image(resized_mat, normalize=True)
            img_array = np.array(img, dtype=np.float64) / 255.0  # Rescale to [0,1]

            # Update plots
            im_phantom.set_data(output)
            im_matrix.set_data(img_array)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    generator_model_path = r"C:\Users\welov\PycharmProjects\ECHO_ML\Grant Files\SavedModels\15e1_ECHO_0.1_model.pth"
    baseline = [0,0,0,0,0,0,0,0.0,0.0,0,0,0,0,0]  # Preset or calibrated beforehand
    electrodecount = 15
    emitting = 1
    if emitting != 1:
        imagesize = (electrodecount(electrodecount-1))/2
    else:
        imagesize = electrodecount - 1

    model = load_generator_model(generator_model_path, image_size=64)

    generate_and_plot_realtime(model, baseline, image_size=64, interval=Z, capsize=imagesize)
