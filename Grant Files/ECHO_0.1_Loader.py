import random

import torch
from CGAN_paper_ECHO_1 import Generator
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
from Construction.modulator2 import generate_circulant_matrix_from_measurements, subtract_baseline, build_circulant_matrix, matrix_to_image

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
    generated = generated.squeeze().cpu().numpy()
    generated = (generated + 1) / 2  # Rescale to [0, 1]
    return generated

Z = 0.05  # seconds between updates

def get_live_measurements():
    """
    Replace this function with real sensor input or stream reader.
    Here we mock it with noise for demonstration.
    """

    base_measurements = [0.0, 0.015493959728274387, 0.019099822788729015, 0.014451753880056817, 0.03200650769829794, 0.10910799095945978,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.09999999680000293, 0.0, 0.0, 0.0, 0.011489592446405461, 0.08012433128113516, 0.0,
     0.22755665007092052, 0.18605893501301995, 0.0, 0.04982257974501636, 0.0, 0.12665463561911428, 0.054058250548558506,
     0.27808767540193335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2584269552337546, 0.21212120484180363,
     0.23933385408952912, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13837226592537577, 0.0, 0.19767440321842722, 0.0, 0.0,
     0.43333334139123114, 0.0, 0.0, 0.021289295156283417, 0.09669424265857918, 0.3196472766711912, 0.3360231553113683,
     0.0, 0.3038305903151268, 0.0, 0.34920636400833205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0317148336537566, 0.0,
     0.0, 0.07637998571242055, 0.07547415045991723, 0.0, 0.0, 0.09859155122107244, 0.0, 0.0, 0.020750744473118976,
     0.02926870428458639, 0.0, 0.07207766607914046, 0.014599459371011814, 0.30689228558551507, 0.0, 0.0, 0.0,
     0.038186947913891545, 0.0, 0.16122308227212823, 0.0, 0.0, 0.0, 0.0, 0.14829914866839122, 0.0, 0.0, 0.0,
     0.012763288175832721, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    baseline = [0.5306680328559553, 0.12469366609178094, 0.030947645277806163, 0.008660060239890688, 0.0044735093472122525, 0.0025854571210364996, 0.0013541187371777993, 0.04420471007036209, 0.0016003864077383176, 0.0011899402867909872, 0.0018466540928920236, 0.0038167955518490226, 0.0047197770687020454, 0.0004307916947013865, 0.44677206400037, 0.09936873559046464, 0.025980890704407428, 0.010712290771265818, 0.006443650768327848, 0.0050481338891066735, 0.01649993367867416, 0.034353991674662396, 0.005746245571594408, 0.003077992473249906, 0.005048133891209464, 0.006689918462836765, 0.0025651115573251046, 0.4203391224699948, 0.08467474113712967, 0.01966001816976469, 0.006115293872814537, 0.0030779924585361065, 0.004022372017336379, 0.004843264072993715, 0.026719691879830913, 0.007388029863193535, 0.0034477473674396422, 0.003980973944189586, 0.0019083977450326806, 0.3939882599638681, 0.07424939414167604, 0.017443608668999723, 0.0057048477233637915, 0.005048133903125293, 0.002831724781554343, 0.0048432640110632336, 0.02598088867465587, 0.01247756096244406, 0.004514907244433819, 0.002072576197404946, 0.37034640712585887, 0.06858523180407067, 0.015309288353511488, 0.008577970773405823, 0.005130223115701625, 0.0025033678814809154, 0.0033242601348172085, 0.04330172143962806, 0.004350728786812058, 0.001744219293127583, 0.34957769065556954, 0.06004794311457155, 0.013995860338921567, 0.0076749891903885245, 0.004883955438837198, 0.004063063172069337, 0.004637687810215903, 0.041413668442442125, 0.0019904869711604156, 0.32807018961433826, 0.02918237172655133, 0.010712290798398895, 0.005704847726401556, 0.0036526170497224245, 0.004719777056319742, 0.009686528387127274, 0.011594927235370433, 0.13310747369547823, 0.025077908487822185, 0.007182453843793395, 0.0041451523977299246, 0.006607829229931375, 0.006854096943723392, 0.005109877650595809, 0.4050703532343321, 0.06866731858008965, 0.017607786675057423, 0.014734663118656021, 0.017771964823147003, 0.10222152036031523, 0.38618973513916566, 0.06300315816573734, 0.031973406574675775, 0.01949583933452867, 0.017259085066213556, 0.3719061226210002, 0.1597044525600512, 0.06382405111277832, 0.004206896120771068, 1.004412208780186, 0.36763744541327714, 0.002072576197404946, 0.8986791638563045, 0.004288985350170561, 0.005520323805193499]


    #changed = (np.array(baseline) - np.array(base_measurements))/np.array(baseline)
    #print(changed)

    noise = np.random.normal(0, 0.01, size=len(base_measurements))
    measurements = np.array(np.array(base_measurements)-np.array(baseline)) + noise
    return measurements.tolist()


def generate_and_plot_realtime(model, baseline, image_size=128, interval=2, capsize=14, device=None):
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
            normalized = measurements
            #normalized = subtract_baseline(measurements, baseline)
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
    generator_model_path = r"C:\Users\welov\PycharmProjects\ECHO_ML\Grant Files\SavedModels\15e15_ECHO_0.1_model.pth"
    baseline = [0]*105

    electrodecount = 15
    emitting = 15
    if emitting != 1:
        imagesize = int((electrodecount*(electrodecount-1))/2)
    else:
        imagesize = electrodecount - 1

    model = load_generator_model(generator_model_path, image_size=64)

    generate_and_plot_realtime(model, baseline, image_size=64, interval=Z, capsize=imagesize)
