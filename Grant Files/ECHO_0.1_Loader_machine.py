import random

import torch
from CGAN_paper_ECHO_1 import Generator
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
from Construction.modulator2 import generate_circulant_matrix_from_measurements, subtract_baseline, build_circulant_matrix, matrix_to_image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from Construction.phantom_generator_newnewattempt import perform_scan_on_phantoms
from functools import partial

follower_radius = None
click_state = 0
circle_radius = circle_radius = random.uniform(0.5, 1.5)

circle_follower = None

def functiontimer(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds of function {func.__name__}")
        return result
    return wrapper


def load_generator_model(model_path="generator_model.pth", image_size=128, device=None):
    """
    Loads a trained generator model from file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator(image_size=image_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_image_from_capacitance(model, capacitance_array, image_size=128, device=None):
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
    generated = generated[0, 0].cpu().numpy()  # Explicitly select [batch, channel]
    generated = np.squeeze(generated)  # This removes extra dims like (1, 64, 64)
    return generated


# Processing pipeline (mock for now)
@functiontimer
def process_phantom(phantom_list, model):
    phantom = phantom_list  # already in (x, y, r) form


    # Generate capacitance matrix
    cap_values = perform_scan_on_phantoms(phantom)  # ← your implementation
    print(cap_values)
    print()

    cap_matrix = build_circulant_matrix(np.array(cap_values))

    cap_matrix_norm = (cap_matrix - np.min(cap_matrix)) / (np.max(cap_matrix) - np.min(cap_matrix) + 1e-8)
    cap_img.set_data(cap_matrix_norm)

    # Normalize and update plot
    cap_img.set_data(cap_matrix)

    # Generate reconstruction
    recon = generate_image_from_capacitance(model, cap_matrix,image_size=64)
    recon_img.set_data(np.fliplr(np.rot90(recon, 2)))

    fig.canvas.draw_idle()

generator_model_path = r"C:\Users\welov\PycharmProjects\ECHO_ML\Grant Files\SavedModels\15e15_ECHO_0.1_model.pth"
model = load_generator_model(generator_model_path, image_size=64)
# Set up figure
fig, (ax_input, ax_cap, ax_recon) = plt.subplots(1, 3, figsize=(21, 8))

# Input plot
circle_radius = random.uniform(0.5, 1.5) #1.06
circle = Circle((0, 0), circle_radius, color='blue', alpha=0.5)

ax_input.add_patch(circle)
boundary_circle_input = Circle((0, 0), 5, color='black', fill=False, linewidth=1.5)
ax_input.add_patch(boundary_circle_input)
ax_input.set_title("Input")
ax_input.set_xlim(-5, 5)
ax_input.set_ylim(-5, 5)
ax_input.set_aspect('equal')

# Capacitance Matrix plot
cap_img = ax_cap.imshow(np.zeros((64, 64)), cmap='viridis', vmin=0, vmax=1, extent=(-5, 5, -5, 5))
ax_cap.set_title("Capacitance Matrix")
ax_cap.set_xlim(-5, 5)
ax_cap.set_ylim(-5, 5)
ax_cap.set_aspect('equal')

# Add colorbar below
cbar = fig.colorbar(cap_img, ax=ax_cap, orientation='horizontal', pad=0.1, fraction=0.046)
cbar.set_label("Normalized Capacitance")

ax_cap.set_xlabel("Excitation Electrodes")
ax_cap.set_ylabel("Received Values")

# Reconstruction plot
recon_img = ax_recon.imshow(np.zeros((64, 64)), cmap='gnuplot', vmin=0, vmax=1, extent=(-5, 5, -5, 5))
boundary_circle_recon = Circle((0, 0), 5, color='white', fill=False, linewidth=1.5)
ax_recon.add_patch(boundary_circle_recon)
ax_recon.set_title("Reconstruction")
ax_recon.set_xlim(-5, 5)
ax_recon.set_ylim(-5, 5)
ax_recon.set_aspect('equal')



def on_mouse_move(model, event):
    global click_state, circle_follower
    if event.inaxes != ax_input:
        return

    x, y = event.xdata, event.ydata

    if click_state == 0:
        circle.center = (x, y)
        process_phantom([(x, y, circle_radius)], model)

    elif click_state == 1 and circle_follower is not None:
        circle_follower.center = (x, y)
        locked_x, locked_y = circle.center
        process_phantom([(locked_x, locked_y, circle_radius), (x, y, follower_radius)], model)
def on_click(event):
    global click_state, circle_follower, circle_radius, follower_radius

    if event.inaxes != ax_input:
        return

    click_state += 1

    if click_state == 1:
        # Lock first circle and create second with radius
        follower_radius = circle_radius = 1.06

        circle_follower = Circle((0, 0), follower_radius, color='red', alpha=0.4)
        ax_input.add_patch(circle_follower)

    elif click_state == 2:
        # Lock both circles in place
        pass

    elif click_state == 3:
        # Reset everything
        click_state = 0
        circle_radius = circle_radius = random.uniform(0.5, 1.5)

        follower_radius = None

        if circle_follower:
            circle_follower.remove()
            circle_follower = None

        cap_img.set_data(np.zeros((64, 64)))
        recon_img.set_data(np.zeros((64, 64)))
        fig.canvas.draw_idle()





# Connect interaction
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', partial(on_mouse_move, model))
plt.show()
