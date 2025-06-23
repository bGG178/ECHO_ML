import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from Construction.modulator2 import build_circulant_matrix
from Construction.phantom_generator_newnewattempt import perform_scan_on_phantoms
from CGAN_paper_ECHO_2 import Generator
import psutil, time

# Toggle between circle and square shapes
USE_SQUARE_SHAPE = False
x_stretch =random.uniform(1,1) #Keep 1 for circle
y_stretch = random.uniform(1,1) #Keep 1 for circle
rotation = random.uniform(0, 360)  # Random rotation angle in degrees

# Shape state
click_state = 0
circle_radius = random.uniform(1, 2)
follower_radius = None
shape = None
follower_shape = None

# Add correlation and mass functions
def imagecorrelation(realarr, fakearr):
    realarr = realarr.flatten()
    fakearr = fakearr.flatten()
    realarr = (realarr - np.mean(realarr)) / (np.std(realarr) + 1e-8)
    fakearr = (fakearr - np.mean(fakearr)) / (np.std(fakearr) + 1e-8)
    correlation = np.corrcoef(realarr, fakearr)[0, 1]
    return max(0, correlation * 100)  # keep it in 0–100 range


def MassCalculator(img):
    """
    Calculates the mass of the image by summing all pixel values.
    """
    return float(torch.sum(torch.tensor(img)))


def generate_ground_truth_image(phantom_list, image_size=64):
    """
    Generate a ground truth binary image based on phantom shape parameters.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(-5, 5, image_size),
        np.linspace(-5, 5, image_size)
    )
    image = np.zeros((image_size, image_size), dtype=np.float32)

    for shape_type, _, x, y, r, xfact, yfact, rotation in phantom_list:
        # Translate grid to shape center
        dx = grid_x - x
        dy = grid_y - y

        if shape_type == 'circle':
            mask = (dx ** 2 + dy ** 2) <= r ** 2

        elif shape_type == 'square':
            mask = (np.abs(dx) <= r) & (np.abs(dy) <= r)

        elif shape_type == 'oval':
            # Convert rotation to radians and apply negative for reverse transform
            theta = -np.deg2rad(rotation)

            # Rotate the coordinate system around the center
            dx_rot = dx * np.cos(theta) - dy * np.sin(theta)
            dy_rot = dx * np.sin(theta) + dy * np.cos(theta)

            # Scale coordinates
            dx_scaled = dx_rot / (r * xfact)
            dy_scaled = dy_rot / (r * yfact)

            # Equation of ellipse: (x/a)^2 + (y/b)^2 <= 1
            mask = dx_scaled ** 2 + dy_scaled ** 2 <= 1

        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        image[mask] = 1.0
    return image

# Define metrics text placeholder
metrics_text = None

def functiontimer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.3f} seconds of function {func.__name__}")
        return result
    return wrapper

def load_generator_model(model_path="generator_model.pth", image_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(image_size=image_size,base_filters=96).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

usagearray = []
usagecount = 0
globalstart = time.time()
timeindex = []

@functiontimer
def generate_image_from_capacitance(model, capacitance_array, image_size=128, device=None):
    global  usagecount

    start = time.time()
    start_cpu = psutil.cpu_times()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    cap_tensor = torch.tensor(capacitance_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cap_tensor = (cap_tensor - 0.5) * 2
    cap_tensor = F.interpolate(cap_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)
    cap_tensor = cap_tensor.to(device)

    with torch.no_grad():
        generated = model(cap_tensor)

    generated = generated[0, 0].cpu().numpy()
    generated = np.squeeze(generated)
    end = time.time()
    end_cpu = psutil.cpu_times()

    cpu_usage = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    elapsed = end - start

    print(f"CPU time: {cpu_usage:.15f}s, Wall time: {elapsed:.5f}s")
    usagecount += 1
    timeindex.append(elapsed)

    if cpu_usage !=0:
        usagearray.append(cpu_usage*4.06)

    return generated


mlossarr = []
icarray = []


def process_phantom(phantom_list, model):
    global metrics_text
    global mlossarr
    global icarray

    #print(phantom_list)

    cap_values = perform_scan_on_phantoms(phantom_list)
    cap_matrix = build_circulant_matrix(np.array(cap_values))
    cap_matrix_norm = (cap_matrix - np.min(cap_matrix)) / (np.max(cap_matrix) - np.min(cap_matrix) + 1e-8)
    cap_img.set_data(cap_matrix_norm)

    # Generate reconstruction
    recon = generate_image_from_capacitance(model, cap_matrix, image_size=64)
    recon_img.set_data(np.fliplr(np.rot90(recon, 2)))

    # Generate ground truth image
    real = generate_ground_truth_image(phantom_list, image_size=64)

    # Compute metrics
    corr = imagecorrelation(real, recon)
    real = np.clip(real, 0, 1)
    recon = np.clip(recon, 0, 1)
    mass_real = MassCalculator(real)
    mass_recon = MassCalculator(recon)

    # Update metrics text
    if metrics_text:
        metrics_text.remove()
    mloss = abs(mass_recon - mass_real) / mass_real * 100

    mlossarr.append(mloss)
    icarray.append(corr)
    metrics_text = ax_recon.text(
        0.05, 0.95,
        f"Correlation: {corr:.2f}%\nMass Error: {(mloss):.2f}%",
        transform=ax_recon.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.4)
    )

    truth_img.set_data(np.fliplr(np.rot90(real, 2)))


    fig.canvas.draw_idle()



active_phantoms = []  # Global list

def on_mouse_move(model, event):
    global active_phantoms

    if event.inaxes != ax_input:
        return

    x, y = event.xdata, event.ydata

    if click_state == 0:
        if USE_SQUARE_SHAPE:
            shape.set_xy((x - circle_radius, y - circle_radius))
        else:
            shape.center = (x, y)

        active_phantoms = [
            ('square' if USE_SQUARE_SHAPE else 'oval', shape, x, y, circle_radius, x_stretch, y_stretch, rotation)
        ]

    elif click_state == 1 and follower_shape is not None:
        if USE_SQUARE_SHAPE:
            follower_shape.set_xy((x - follower_radius, y - follower_radius))
        else:
            follower_shape.center = (x, y)

        # Update active_phantoms to include both
        x1, y1 = (shape.center if not USE_SQUARE_SHAPE else (shape.get_x() + circle_radius, shape.get_y() + circle_radius))
        active_phantoms = [
            ('square' if USE_SQUARE_SHAPE else 'oval', shape, x1, y1, circle_radius, x_stretch, y_stretch, rotation),
            ('square' if USE_SQUARE_SHAPE else 'oval', follower_shape, x, y, follower_radius, x_stretch, y_stretch, rotation)
        ]

    process_phantom(active_phantoms, model)


def on_click(event):
    global click_state, shape, follower_shape, circle_radius, follower_radius

    if event.inaxes != ax_input:
        return

    click_state += 1

    if click_state == 1:
        follower_radius = random.uniform(0.5, 1.5)
        if USE_SQUARE_SHAPE:
            follower_shape = Rectangle((-follower_radius, -follower_radius), 2 * follower_radius, 2 * follower_radius,
                                       color='red', alpha=0.4)
        else:
            follower_shape = Circle((0, 0), follower_radius, color='red', alpha=0.4)
        ax_input.add_patch(follower_shape)

    elif click_state == 2:
        pass  # Lock both shapes

    elif click_state == 3:
        click_state = 0
        circle_radius = random.uniform(0.5, 1.5)
        follower_radius = None

        if follower_shape:
            follower_shape.remove()
            follower_shape = None

        cap_img.set_data(np.zeros((64, 64)))
        recon_img.set_data(np.zeros((64, 64)))
        fig.canvas.draw_idle()

# Load model
generator_model_path = r"C:\Users\bower\PycharmProjects\ECHO_ML\Grant Files\SavedModels\15e15_ECHO_0.2.3.1_model.pth"
model = load_generator_model(generator_model_path, image_size=64)

# Set up figure
fig, (ax_input, ax_truth,ax_cap, ax_recon) = plt.subplots(1, 4, figsize=(28, 8))

# Input plot
if USE_SQUARE_SHAPE:
    shape = Rectangle((-circle_radius, -circle_radius), 2 * circle_radius, 2 * circle_radius, color='blue', alpha=0.5)
else:
    shape = Circle((0, 0), circle_radius, color='blue', alpha=0.5)
ax_input.add_patch(shape)
boundary_circle_input = Circle((0, 0), 5, color='black', fill=False, linewidth=1.5)
ax_input.add_patch(boundary_circle_input)
ax_input.set_title("Input", fontsize=20)
ax_input.set_xlim(-5, 5)
ax_input.set_ylim(-5, 5)
ax_input.set_aspect('equal')
ax_input.set_xlabel("Units", fontsize=14)
ax_input.set_ylabel("Units", fontsize=14)

# Capacitance Matrix plot
cap_img = ax_cap.imshow(np.zeros((105, 105)), cmap='viridis', vmin=0, vmax=1, extent=(0, 105, 0, 105))
ax_cap.set_title("Capacitance Matrix", fontsize=20)
ax_cap.set_xlim(0, 105)
ax_cap.set_ylim(0, 105)
ax_cap.set_xlabel("Excitation Electrode Values", fontsize=16)
ax_cap.set_ylabel("Rotated Values", fontsize=16)
ax_cap.set_aspect('equal')

# Save original position before adding colorbar
pos_cap = ax_cap.get_position()

# Add colorbar for Capacitance Matrix
cap_colorbar = fig.colorbar(cap_img, ax=ax_cap, fraction=0.2, pad=0.04, orientation='horizontal')
cap_colorbar.set_label('Sensor Returned Value', fontsize=14)

# Reset ax_cap position and shift it up a bit to make room for colorbar below
ax_cap.set_position(pos_cap)
pos_cap = ax_cap.get_position()
ax_cap.set_position([pos_cap.x0, pos_cap.y0 + 0.127, pos_cap.width, pos_cap.height])

# Reconstruction plot
recon_img = ax_recon.imshow(np.zeros((64, 64)), cmap='gnuplot', vmin=0, vmax=1, extent=(-5, 5, -5, 5))
boundary_circle_recon = Circle((0, 0), 5, color='white', fill=False, linewidth=1.5)
ax_recon.add_patch(boundary_circle_recon)
ax_recon.set_title("Reconstruction", fontsize=20)
ax_recon.set_xlim(-5, 5)
ax_recon.set_ylim(-5, 5)
ax_recon.set_aspect('equal')
ax_recon.set_xlabel("Units", fontsize=14)
ax_recon.set_ylabel("Units", fontsize=14)

# Save original position before adding colorbar
pos_recon = ax_recon.get_position()

# Add colorbar for Reconstruction
recon_colorbar = fig.colorbar(recon_img, ax=ax_recon, fraction=0.2, pad=0.04, orientation='horizontal')
recon_colorbar.set_label('Reconstruction Confidence', fontsize=14)

# Reset ax_recon position and shift it up a bit to make room for colorbar below
ax_recon.set_position(pos_recon)
pos_recon = ax_recon.get_position()
ax_recon.set_position([pos_recon.x0, pos_recon.y0 + 0.127, pos_recon.width, pos_recon.height])

# Ground truth image
truth_img = ax_truth.imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1, extent=(-5, 5, -5, 5))
boundary_circle_truth = Circle((0, 0), 5, color='white', fill=False, linewidth=1.5)
ax_truth.add_patch(boundary_circle_truth)
ax_truth.set_title("Ground Truth", fontsize=20)
ax_truth.set_xlim(-5, 5)
ax_truth.set_ylim(-5, 5)
ax_truth.set_aspect('equal')
ax_truth.set_xlabel("Units", fontsize=14)
ax_truth.set_ylabel("Units", fontsize=14)

# Connect interaction
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', partial(on_mouse_move, model))
plt.show()

print(str(mlossarr))
python_list = [float(x) for x in icarray]
print(python_list)
#print(usagearray)
#print(usagecount)
#print(globalstart- time.time())
print(timeindex)