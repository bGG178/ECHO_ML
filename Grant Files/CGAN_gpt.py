"""
To Do:
Double check ground truth image and reconstruction are set up properly
Increase reconstruction accuracy
Add more comments
Refine losses
Have the algorithm guess the area based on capacitance data

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from glob import glob
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity  as ssim  # Add this import

# --- Hyperparameters ---
BATCH_SIZE = 64  # Number of samples per batch
IMAGE_SIZE_X = 15  # Width of images
IMAGE_SIZE_Y = 16  # Height of images
LATENT_DIM = 100  # Dimensionality of noise vector for Generator
EPOCHS = 10  # Number of training epochs
LEARNING_RATE_D = 0.00001  # Learning rate for discriminator
LEARNING_RATE_G = 0.00004  # Learning rate for generator
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Training on GPU if available

def save_generated_images(generator, epoch, measurement_sample):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(measurement_sample.size(0), LATENT_DIM, device=DEVICE)
        gen_imgs = generator(z, measurement_sample.to(DEVICE))
        gen_imgs = gen_imgs.cpu().numpy()

    os.makedirs("outputs", exist_ok=True)
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axs[i].imshow(gen_imgs[i][0], cmap='gray')
        axs[i].axis('off')
    plt.savefig(f"outputs/epoch_{epoch+1:03d}.png")
    plt.close()

def calculate_mse(img1, img2):
    # Ensure img1 and img2 have the same shape
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    return np.mean((img1 - img2) ** 2)

def calculate_ssim(img1, img2):
    # Ensure img1 and img2 have the same shape
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def reconstruct_image(generator, measurement_sample, ground_truth_image=None, output_path="reconstructed_image.png"):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(measurement_sample.size(0), LATENT_DIM, device=DEVICE)
        measurement_sample = measurement_sample.view(-1, 1)  # Ensure measurement_sample has the correct dimensions
        gen_imgs = generator(z, measurement_sample.to(DEVICE))
        gen_imgs = gen_imgs.cpu().numpy()

    if measurement_sample.size(0) == 1:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(gen_imgs[0][0], cmap='gray')
        ax.axis('off')
    else:
        fig, axs = plt.subplots(1, measurement_sample.size(0), figsize=(15, 3))
        for i in range(measurement_sample.size(0)):
            axs[i].imshow(gen_imgs[i][0], cmap='gray')
            axs[i].axis('off')
    plt.savefig(output_path)
    plt.close()

    if ground_truth_image is not None:
        ground_truth_image = ground_truth_image.cpu().numpy()
        mse = calculate_mse(gen_imgs[0][0], ground_truth_image[0][0])
        ssim_value = calculate_ssim(gen_imgs[0][0], ground_truth_image[0][0])
        print(f"MSE: {mse:.4f}, SSIM: {ssim_value:.4f}")

# ---------------------------
# Dataset for ECT from folders
# ---------------------------
class FolderECTDataset(Dataset):
    def __init__(self, root_dir="C:/Users/welov/PycharmProjects/ECHO_ML/DATA/MLTriangleTrainingData"):
        self.sample_paths = []
        for folder in glob(os.path.join(root_dir, "*")):
            image_files = glob(os.path.join(folder, "*.jpg"))
            if image_files:
                area = int(os.path.basename(folder))
                for img_file in image_files:
                    self.sample_paths.append((img_file, area))

        # Remove one image at random for ground truth testing
        self.ground_truth_image_path = Path(r"C:/Users/welov/PycharmProjects/ECHO_ML/DATA/MLTriangleTrainingData/4/4.960.jpg")
        # Save the ground truth image as "base_image.png"
        ground_truth_image = Image.open(self.ground_truth_image_path).convert("L")
        ground_truth_image.save("base_image.png")

        print(f"Ground truth image path: {self.ground_truth_image_path}")


        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_SIZE_Y, IMAGE_SIZE_X)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        img_path, area = self.sample_paths[idx]
        capacitance_data = Image.open(img_path).convert("RGB")
        capacitance_data = self.transform(capacitance_data)

        area = torch.tensor(area, dtype=torch.float32)
        return capacitance_data, area

# ---------------------------
# Generator Model Definition
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + 1, 256),  # Increased number of neurons
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, IMAGE_SIZE_Y * IMAGE_SIZE_X),
            nn.Tanh()
        )

    def forward(self, z, measurement):
        measurement = measurement.view(-1, 1)
        x = torch.cat([z, measurement], dim=1)
        img = self.model(x)
        img = img.view(-1, 1, IMAGE_SIZE_Y, IMAGE_SIZE_X)
        return img

# -------------------------------
# Discriminator Model Definition
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear((IMAGE_SIZE_Y * IMAGE_SIZE_X) + 1, 512),  # Increased number of neurons
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, measurement):
        if img.ndimension() == 4:
            img = img.view(img.size(0), -1)
        elif img.ndimension() == 2:
            img = img.view(img.size(0), -1)
        img_flat = img
        measurement_flat = measurement.view(measurement.size(0), -1)
        x = torch.cat([img_flat, measurement_flat], dim=1)
        validity = self.model(x)
        return validity

# -----------------------------
# Training Function for CGAN
# -----------------------------
def train():
    dataset = FolderECTDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        for i, (real_imgs, measurement) in enumerate(dataloader):  # Corrected order
            batch_size = measurement.size(0)
            real_imgs = real_imgs.to(DEVICE)
            measurement = measurement.to(DEVICE)

            valid = torch.ones((batch_size, 1), device=DEVICE)
            fake = torch.zeros((batch_size, 1), device=DEVICE)

            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z, measurement)

            optimizer_G.zero_grad()
            g_loss = loss_fn(discriminator(gen_imgs, measurement), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = loss_fn(discriminator(real_imgs, measurement), valid)
            fake_loss = loss_fn(discriminator(gen_imgs.detach(), measurement), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        save_generated_images(generator, epoch, measurement[:5])

    torch.save(generator.state_dict(), "generator_cgan_ect.pth")
    torch.save(discriminator.state_dict(), "discriminator_cgan_ect.pth")

if __name__ == "__main__":
    train()
    # Load the trained generator
    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load("generator_cgan_ect.pth"))
    generator.eval()

    # Example measurement sample for reconstruction
    example_measurement = torch.tensor([0.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Load ground truth image for comparison (example)
    dataset = FolderECTDataset()
    ground_truth_image_path = dataset.ground_truth_image_path
    if os.path.exists(ground_truth_image_path):
        ground_truth_image = Image.open(ground_truth_image_path).convert("L")
        ground_truth_image = transforms.ToTensor()(ground_truth_image).unsqueeze(0).to(DEVICE)
        reconstruct_image(generator, example_measurement, ground_truth_image)
    else:
        print(f"Ground truth image not found at {ground_truth_image_path}")
