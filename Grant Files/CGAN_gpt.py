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

# --- Hyperparameters ---
BATCH_SIZE = 16  # Number of samples per batch
IMAGE_SIZE = 16  # Width/Height of square images
LATENT_DIM = 100  # Dimensionality of noise vector for Generator
EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 2e-4  # Learning rate for optimizers
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

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        img_path, area = self.sample_paths[idx]
        capacitance_data = Image.open(img_path).convert("RGB")
        capacitance_data = self.transform(capacitance_data)

        # Check the shape of capacitance_data
        print(f"capacitance_data shape: {capacitance_data.shape}")

        area = torch.tensor(area, dtype=torch.float32)
        return capacitance_data, area




# ---------------------------
# Generator Model Definition
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256 + LATENT_DIM, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, z, measurement):
        flattened = measurement.view(measurement.size(0),-1)
        x = torch.cat([z, flattened], dim=1)
        img = self.model(x)
        img = img.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        return img

# -------------------------------
# Discriminator Model Definition
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256 + 256, 512),  # Concatenated size of image and measurement
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, measurement):
        # Ensure img is a 4D tensor
        if img.ndimension() == 4:
            img = img.view(img.size(0), -1)  # Flatten to (batch_size, 256)
        elif img.ndimension() == 2:
            img = img.view(img.size(0), -1)  # Ensure it's (batch_size, 256)
        else:
            print(f"Unexpected img dimension: {img.ndimension()}")

        img_flat = img
        measurement_flat = measurement.view(measurement.size(0), -1)  # Flatten measurement to (batch_size, 256)

        print(f"img_flat shape: {img_flat.shape}")  # Check shape of img_flat
        print(f"measurement_flat shape: {measurement_flat.shape}")  # Check shape of measurement_flat

        # Concatenate the flattened image and measurement tensor
        x = torch.cat([img_flat, measurement_flat], dim=1)

        print(f"Concatenated tensor shape: {x.shape}")  # Check shape after concatenation

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

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        for i, (measurement, real_imgs) in enumerate(dataloader):
            batch_size = measurement.size(0)
            real_imgs = real_imgs.to(DEVICE)
            print(f'INITIAL DIM: {real_imgs.ndimension()}')
            measurement = measurement.to(DEVICE)

            # Check the shape of real_imgs
            print(f"real_imgs shape: {real_imgs.shape}")
            if real_imgs.ndimension() != 4:
                print(f"Unexpected real_imgs dimension: {real_imgs.ndimension()}")

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

    print(os.path.exists(Path("C:/Users/welov/PycharmProjects/ECHO_ML/DATA/MLTriangleTrainingData")))
    train()




