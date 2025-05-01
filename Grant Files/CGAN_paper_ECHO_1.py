import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import torch.nn.functional as F
from shapely.geometry import Point
import numpy as np
from Construction import modulator2 as m2
import time

def functiontimer(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds of function {func.__name__}")
        return result
    return wrapper

def build_phantom_from_objects(objects, image_size=64):
    """
    Much faster version of rasterized phantom image creation.
    Vectorized circle filling using numpy broadcasting.
    """
    img = np.zeros((image_size, image_size), dtype=np.float32)

    # Create a meshgrid for pixel coordinates
    x = np.linspace(-5, 5, image_size)
    y = np.linspace(-5, 5, image_size)
    xv, yv = np.meshgrid(x, y)

    for obj in objects:
        cx, cy = obj["center"]
        r = obj["radius"]

        # Compute squared distance from each pixel to the circle center
        mask = (xv - cx)**2 + (yv - cy)**2 <= r**2
        img[mask] = 1.0  # Set all pixels inside the circle to 1

    return img

def show_capacitance_image(ax, cond_input, title="Capacitance Image"):
    """
    Displays the capacitance input image.

    Parameters:
    - ax: Matplotlib axis to plot the image.
    - cond_input: Capacitance input image (2D array).
    - title: Title for the plot.
    """
    ax.imshow(cond_input, cmap="viridis")
    ax.set_title(title, fontsize=6)
    ax.axis("off")

def cutoff(tensr, x, tensorreturn = True):
    """
    Applies a cutoff to the input array.
    Values above x are set to 1, and values below or equal to x are set to 0.

    tensorreturn because sometimes it whines at you if you don't return a tensor

    """
    tensr = torch.tensor(tensr) if not isinstance(tensr, torch.Tensor) else tensr
    result = (tensr > x).float()
    return result if tensorreturn else result.numpy()

def adjust_learning_rate(optimizer, target_loss, current_loss, factor=0.1, min_lr=1e-6, max_lr=1e-2):
    """
    Adjusts the learning rate of the optimizer based on the current loss and target loss.
    """
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        if current_loss > target_loss:
            new_lr = min(current_lr * (1 + factor), max_lr)  # Increase LR
        else:
            new_lr = max(current_lr * (1 - factor), min_lr)  # Decrease LR
        param_group['lr'] = new_lr

def imagecorrelation(realarr,fakearr):
    """
    Accepts the real and fake image, overlays them, and calculates the correlation by measuring the difference in pixels
    Returns a percentage value of correlation
    """
    total=0

    realarr = realarr.flatten()
    fakearr = fakearr.flatten()
    diff = np.abs(realarr - fakearr)
    maximumtotal = len(realarr)

    for value in diff:
        total += value
    correlation = 1- (total / maximumtotal) # 1- because if it is 100% accurate, then each value is 0 or close to. So total ends up being very small.
    return correlation * 100 # Return to be displayed as a %

class Generator(nn.Module): # U-Net Generator
    def __init__(self, image_size=128, base_filters=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 2 + base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters + base_filters, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.image_size = image_size


    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        generated_image = self.final(d1)

        return generated_image.view(-1, 1, self.image_size, self.image_size)  # Reshape to image format

class Discriminator(nn.Module):
    def __init__(self,image_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(128 * (image_size//4) * (image_size//4), 1),
            nn.Sigmoid()
        )

    def forward(self, real_imgs, fake_imgs):
        # Concatenate real and fake images along the channel dimension
        x = torch.cat([real_imgs, fake_imgs], dim=1)
        return self.model(x)


def MassCalculator(img):
    """
    Calculates the mass of the image by summing all pixel values.
    """
    return torch.sum(img)

if __name__ == "__main__":
    #######################################################HYPERPARAMETERS!!!!#############################################################################################
    batch_size = 16
    image_size = 64 #WORKS WELL FOR 32x32, 64x64 and 128x128 need work!
    target_loss = 0.45
    epochs = 50
    electrode_count = 15
    emitting_count = 15
    datafile = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData\npg_3500_15e15_traintest.json"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #Change data to location of the tenk_64 data.json
    #tenk_64.json is a dataset of 10,000 phantoms with capacitance measurements in 64x64 resolution
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"File not found: {datafile}")

    # Load the JSON file
    with open(datafile, 'r') as f:
        combined_data = json.load(f)


    # Split the dataset into training (80%) and testing (20%)
    # This trains the algorithm on 80% of the data and tests it on 20%
    train_size = int(0.8 * len(combined_data))
    train_data = combined_data[:train_size]
    test_data = combined_data[train_size:]

    #Dataloaders
    if 1==1:

        train_real_images = torch.tensor(
            np.array([build_phantom_from_objects(sample["objects"], image_size=image_size) for sample in train_data]),
            dtype=torch.float32
        ).unsqueeze(1)
        train_real_images = (train_real_images - 0.5) * 2  # Normalize to [-1,1]

        train_capacitance_np = np.array(
            [m2.build_circulant_matrix(np.array(sample["measurements"])) for sample in train_data],
            dtype=np.float32  # This makes the conversion faster and more consistent
        )
        train_capacitance = torch.tensor(train_capacitance_np).unsqueeze(1)

        train_capacitance = (train_capacitance - 0.5) * 2

        train_dataset = TensorDataset(train_real_images, train_capacitance)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        test_real_images = torch.tensor(
            np.array([build_phantom_from_objects(sample["objects"], image_size=image_size) for sample in test_data]),
            dtype=torch.float32
        ).unsqueeze(1)
        test_real_images = (test_real_images - 0.5) * 2

        test_capacitance_np = np.array(
            [m2.build_circulant_matrix(np.array(sample["measurements"])) for sample in test_data],
            dtype=np.float32
        )
        test_capacitance = torch.tensor(test_capacitance_np).unsqueeze(1)

        test_capacitance = (test_capacitance - 0.5) * 2

        test_dataset = TensorDataset(test_real_images, test_capacitance)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    ##############

    #Initialize the Generator and Discriminator
    G = Generator(image_size=image_size).to(device)
    D = Discriminator(image_size=image_size).to(device)

    #Initialize the optimization functions
    optimizer_G = optim.Adam(G.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0001)

    #Initialize the scheduler functions
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.9)

    #Initialize the loss functions for the CGAN
    loss_GAN = nn.BCELoss()
    loss_MSE = nn.MSELoss()

    #Initialize trackers for the loss and image correlation
    Dlossarr =[]
    Glossarr =[]
    MSEarr =[]
    ICarr = []

    # Training loop
    for epoch in trange(epochs+1):
        max_batch_size = train_dataloader.batch_size
        valid = torch.ones((max_batch_size, 1), device=device)
        fake = torch.zeros((max_batch_size, 1), device=device)
        for i, (real_imgs, cond_input) in enumerate(train_dataloader):

            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            caps = cond_input.to(device)

            caps_flat = caps.view(caps.size(0), -1)
            #print(i)

            if (epoch == 0) & (i == 0):
                # Calculate the correct dimensions for reshaping
                height = int(caps_flat.size(1) ** 0.5)
                width = height  # Assuming the tensor is square
                if height * width != caps_flat.size(1):
                    raise ValueError("The tensor cannot be reshaped into a square image.")

                # Reshape and display the image
                # This displays the first image in the batch to show what the capacitance data (being fed into the algorithm) looks like
                plt.imshow(caps_flat[0].view(height, width).cpu(), cmap="viridis")
                plt.colorbar()
                plt.title("Capacitance Image")
                #plt.show()


            # valid and fake labels to train the discriminator
            #valid = torch.ones((caps.size(0), 1), device=device)
            #fake = torch.zeros((caps.size(0), 1), device=device)
            caps_reshaped = F.interpolate(caps, size=(image_size, image_size), mode='bilinear', align_corners=False)

            # Generator step
            #Generator trained more frequently than the discriminator
            for _ in range(2):
                #Reset the optimizer
                optimizer_G.zero_grad()
                # Pass the capacitance data to the generator and generate the fake images
                fake_imgs = G(caps_reshaped)
                # Pass the fake images to the discriminator
                g_loss = loss_GAN(D(real_imgs, fake_imgs), valid)
                # Calculate the MSE loss
                g_loss.backward()
                #Step the gradient function of the optimizer
                optimizer_G.step()

            # Discriminator step
            optimizer_D.zero_grad()
            # Pass real and fake images to the discriminator
            real_loss = loss_GAN(D(real_imgs, real_imgs), valid)  # Real images as both inputs
            fake_loss = loss_GAN(D(real_imgs, fake_imgs.detach()), fake)  # Real and fake images
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Mass loss calculations
            fakeimgmass = MassCalculator(fake_imgs)
            realimgmass = MassCalculator(real_imgs)
            mloss = (abs(float(fakeimgmass) - float(realimgmass)) / float(realimgmass)) * 100
            with torch.no_grad():
                ic = imagecorrelation(cutoff(real_imgs[0].cpu().numpy(),0.5,tensorreturn=False), cutoff(fake_imgs[0].cpu().numpy(),0.5,tensorreturn=False))

            # Adjust learning rates dynamically based on the target loss rate
            adjust_learning_rate(optimizer_G, target_loss, g_loss)
            adjust_learning_rate(optimizer_D, target_loss, d_loss)

        # Save the loss and image correlation values for plotting every epoch
        if epoch % 1 == 0:
            Dlossarr.append(d_loss.item())
            Glossarr.append(g_loss.item())
            MSEarr.append(mloss)
            with torch.no_grad():
                ICarr.append(imagecorrelation(real_imgs[0].cpu().numpy(), fake_imgs[0].cpu().numpy()))

        # Visualization
        # No show function, they will pop up after the end of training so you can let it run
        if epoch % 4 == 0:
            with torch.no_grad():
                print(
                    f"[{epoch}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, Mass Error: {mloss:.2f}%, IC%: {ic:.2f}%")
                fig, axs = plt.subplots(3, 8, figsize=(15, 12))
                for i in range(8):
                    axs[0, i].imshow(cutoff(real_imgs[i, 0].cpu(), 0.5), cmap="viridis")
                    axs[0, i].set_title("Real")
                    axs[0, i].axis("off")
                    axs[1, i].imshow((fake_imgs[i, 0].cpu()), cmap="viridis")
                    ic_value = imagecorrelation(
                        cutoff(real_imgs[i, 0].cpu().numpy(), 0.5, tensorreturn=False),
                        cutoff(fake_imgs[i, 0].cpu().numpy(), 0.5, tensorreturn=False)
                    )
                    mlossgraph = (abs(float(MassCalculator(fake_imgs[i, 0])) - float(
                        MassCalculator(real_imgs[i, 0]))) / float(MassCalculator(real_imgs[i, 0]))) * 100
                    axs[1, i].set_title(f"IC: {ic_value:.2f}%, ME: {mlossgraph:.2f}%", fontsize=6)
                    axs[1, i].axis("off")

                    # Add capacitance image
                    show_capacitance_image(axs[2, i], cond_input[i, 0].cpu().numpy(), title="Capacitance Input")
                plt.tight_layout()

        #scheduler step
        scheduler_G.step()
        scheduler_D.step()

    # Save Generator after training

    modeloutput = r"C:\Users\welov\PycharmProjects\ECHO_ML\Grant Files\SavedModels"
    os.makedirs(modeloutput, exist_ok=True)
    output_file = os.path.join(modeloutput, f"{electrode_count}e{emitting_count}_ECHO_0.1_model.pth")
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"File not found: {datafile}")

    torch.save(G.state_dict(), output_file)
    print("Generator model saved.")

    plt.figure(figsize=(15, 5))

    # Plot Glossarr and Dlossarr on the first subplot
    plt.subplot(1, 3, 1)
    plt.plot(Dlossarr, label='Discriminator Loss')
    plt.plot(Glossarr, label='Generator Loss')
    plt.xlabel('Epochs (every 20)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss')

    # Plot MSEarr on the second subplot
    plt.subplot(1, 3, 2)
    plt.plot(MSEarr, label='Mass Error', color='orange')
    plt.xlabel('Epochs (every 20)')
    plt.ylabel('% Error')
    plt.legend()
    plt.title('Mass Error')

    # Plot MSEarr on the second subplot
    plt.subplot(1, 3, 3)
    plt.plot(ICarr, label='Image Correlation', color='blue')
    plt.xlabel('Epochs (every 20)')
    plt.ylabel('% Correlation')
    plt.legend()
    plt.title('Image Correlation')

    plt.tight_layout()
    plt.show()






    # Testing loop
    with torch.no_grad():
        for i, (real_imgs, cond_input) in enumerate(test_dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            caps = cond_input.to(device)

            caps_flat = caps.view(caps.size(0), -1)
            valid = torch.ones((caps.size(0), 1), device=device)
            fake = torch.zeros((caps.size(0), 1), device=device)
            caps_reshaped = F.interpolate(caps, size=(image_size, image_size), mode='bilinear', align_corners=False)


            fake_imgs = G(caps_reshaped)

            # Mass loss
            fakeimgmass = MassCalculator(fake_imgs)
            realimgmass = MassCalculator(real_imgs)
            mloss = (abs(float(fakeimgmass) - float(realimgmass)) / float(realimgmass)) * 100
            with torch.no_grad():
                ic = imagecorrelation(cutoff(real_imgs[0].cpu().numpy(),0.5,tensorreturn=False), cutoff(fake_imgs[0].cpu().numpy(),0.5,tensorreturn=False))

            # Adjust learning rates
            adjust_learning_rate(optimizer_G, target_loss, g_loss)
            adjust_learning_rate(optimizer_D, target_loss, d_loss)




            Dlossarr.append(d_loss.item())
            Glossarr.append(g_loss.item())
            MSEarr.append(mloss)

            with torch.no_grad():
                print(
                    f"[{epoch}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, Mass Error: {mloss:.2f}%, IC%: {ic:.2f}%")
                fig, axs = plt.subplots(3, 8, figsize=(15, 12))  # Add a third row for capacitance images
                for i in range(8):
                    axs[0, i].imshow(cutoff(real_imgs[i, 0].cpu(), 0.5), cmap="viridis")
                    axs[0, i].set_title("Test Real Case")
                    axs[0, i].axis("off")

                    axs[1, i].imshow((fake_imgs[i, 0].cpu()), cmap="viridis")
                    ic_value = imagecorrelation(
                        cutoff(real_imgs[i, 0].cpu().numpy(), 0.5, tensorreturn=False),
                        cutoff(fake_imgs[i, 0].cpu().numpy(), 0.5, tensorreturn=False)
                    )
                    mlossgraph = (abs(float(MassCalculator(fake_imgs[i, 0])) - float(
                        MassCalculator(real_imgs[i, 0]))) / float(MassCalculator(real_imgs[i, 0]))) * 100
                    axs[1, i].set_title(f"IC: {ic_value:.2f}%, ME: {mlossgraph:.2f}%", fontsize=6)
                    axs[1, i].axis("off")

                    # Add capacitance image
                    show_capacitance_image(axs[2, i], cond_input[i, 0].cpu().numpy(), title="Capacitance Input")

                plt.tight_layout()
                plt.show()

