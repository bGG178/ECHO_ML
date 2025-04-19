import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, input_dim=32, img_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_size * img_size),
            nn.Tanh()
        )
        self.img_size = img_size

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1, self.img_size, self.img_size)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, img_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, cap_img, recon_img):
        x = torch.cat([cap_img, recon_img], dim=1)
        return self.model(x)

# --- Mass Estimator ---
class MassEstimator(nn.Module):
    def __init__(self, img_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1)
        )

    def forward(self, x):
        return self.model(x)

# --- Complex Flow Patterns ---
def generate_complex_pattern(img_size):
    pattern_type = random.choice(["gaussian", "vertical", "radial", "swirl"])
    img = np.zeros((img_size, img_size), dtype=np.float32)
    yy, xx = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size), indexing="ij")

    if pattern_type == "gaussian":
        for _ in range(np.random.randint(1, 4)):
            cx, cy = np.random.uniform(-0.5, 0.5, size=2)
            sigma = np.random.uniform(0.1, 0.3)
            blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            img += blob

    elif pattern_type == "vertical":
        gradient = np.linspace(0, 1, img_size).reshape(-1, 1)
        img = np.tile(gradient, (1, img_size))

    elif pattern_type == "radial":
        r = np.sqrt(xx**2 + yy**2)
        img = 1 - np.clip(r, 0, 1)

    elif pattern_type == "swirl":
        angle = np.arctan2(yy, xx)
        r = np.sqrt(xx**2 + yy**2)
        img = 0.5 + 0.5 * np.sin(10 * angle + 5 * r)

    img = img - img.min()
    img = img / img.max()
    img *= np.random.uniform(0.5, 1.0)
    return img.astype(np.float32)

# --- Simulated Data ---
def generate_sample(batch_size=16, img_size=32):
    images, caps, masses = [], [], []
    for _ in range(batch_size):
        img = generate_complex_pattern(img_size)
        cap_vector = np.sum(img, axis=1) / img_size
        cap_vector += np.random.normal(0, 0.01, size=cap_vector.shape)  # Add noise
        mass = np.sum(img)
        images.append(img)
        caps.append(cap_vector)
        masses.append(mass)
    caps = np.expand_dims(np.array(caps), axis=1)
    images = np.expand_dims(np.array(images), axis=1)
    return (
        torch.tensor(caps, dtype=torch.float32),
        torch.tensor(images, dtype=torch.float32),
        torch.tensor(masses, dtype=torch.float32).view(-1, 1)
    )

# --- Initialize ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32
input_dim = img_size

G = Generator(input_dim=input_dim, img_size=img_size).to(device)
D = Discriminator(img_size=img_size).to(device)
M = MassEstimator(img_size=img_size).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=2e-4)
optimizer_D = optim.Adam(D.parameters(), lr=2e-4)
optimizer_M = optim.Adam(M.parameters(), lr=1e-4)

loss_GAN = nn.BCELoss()
loss_MSE = nn.MSELoss()

# --- Training Loop ---
epochs = 10001
for epoch in trange(epochs):
    caps, real_imgs, masses = generate_sample(batch_size=16, img_size=img_size)
    caps, real_imgs, masses = caps.to(device), real_imgs.to(device), masses.to(device)

    caps_flat = caps.view(caps.size(0), -1)
    valid = torch.ones((caps.size(0), 1), device=device)
    fake = torch.zeros((caps.size(0), 1), device=device)

    # Generator step
    for _ in range(3):
        optimizer_G.zero_grad()
        fake_imgs = G(caps_flat)
        cap_imgs = caps.view(caps.size(0), 1, img_size, 1).expand(-1, 1, img_size, img_size)
        g_loss = loss_GAN(D(cap_imgs, fake_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

    # Discriminator step
    optimizer_D.zero_grad()
    real_loss = loss_GAN(D(cap_imgs, real_imgs), valid)
    fake_loss = loss_GAN(D(cap_imgs, fake_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # Mass estimator step
    optimizer_M.zero_grad()
    mass_pred = M(fake_imgs.detach())
    m_loss = loss_MSE(mass_pred, masses)
    m_loss.backward()
    optimizer_M.step()

    # Visualization
    if epoch % 1000 == 0:
        print(f"[{epoch}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, Mass_MSE: {m_loss.item():.4f}")
        with torch.no_grad():
            fig, axs = plt.subplots(2, 4, figsize=(10, 5))
            for i in range(4):
                axs[0, i].imshow(real_imgs[i, 0].cpu(), cmap="viridis")
                axs[0, i].set_title("Real")
                axs[0, i].axis("off")
                axs[1, i].imshow(fake_imgs[i, 0].cpu(), cmap="viridis")
                axs[1, i].set_title("Fake")
                axs[1, i].axis("off")
            plt.tight_layout()
            plt.show()
