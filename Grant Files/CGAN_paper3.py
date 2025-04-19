import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

def cutoff(tensr, x, tensorreturn = True):
    """
    Applies a cutoff to the input array.
    Values above x are set to 1, and values below or equal to x are set to 0.

    Parameters:
        array (numpy.ndarray): Input array.
        x (float): Cutoff value.

    Returns:
        numpy.ndarray: Array with values set to 1 or 0 based on the cutoff.
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
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (image_size//4) * (image_size//4), 1),
            nn.Sigmoid()
        )
    def forward(self, cap_img, recon_img):
        x = torch.cat([cap_img, recon_img], dim=1)
        return self.model(x)

class MassEstimator(nn.Module):
    def __init__(self, image_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (image_size//4) * (image_size//4), 1)
        )

    def forward(self, x):
        return self.model(x)

def MassCalculator(img):
    """
    Calculates the mass of the image by summing all pixel values.
    """
    return torch.sum(img)


# --- Complex Flow Patterns ---
def generate_complex_pattern(image_size):
    pattern_type = random.choice(["gaussian", "vertical", "radial", "swirl"])
    img = np.zeros((image_size, image_size), dtype=np.float32)
    yy, xx = np.meshgrid(np.linspace(-1, 1, image_size), np.linspace(-1, 1, image_size), indexing="ij")

    if pattern_type == "gaussian":
        for _ in range(np.random.randint(1, 4)):
            cx, cy = np.random.uniform(-0.5, 0.5, size=2)
            sigma = np.random.uniform(0.1, 0.3)
            blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            img += blob

    elif pattern_type == "vertical":
        gradient = np.linspace(0, 1, image_size).reshape(-1, 1)
        img = np.tile(gradient, (1, image_size))

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


def generate_sample(batch_size=16, image_size=32):
    images, caps, masses = [], [], []
    for _ in range(batch_size):
        img = generate_complex_pattern(image_size)
        cap_vector = np.sum(img, axis=1) / image_size
        #cap_vector += np.random.normal(0, 0.01, size=cap_vector.shape)  # Add noise
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 32 #WORKS WELL FOR 32x32, 64x64 and 128x128 need work!
target_loss = 0.45
epochs = 4000

G = Generator(image_size=image_size).to(device)
D = Discriminator(image_size=image_size).to(device)
M = MassEstimator(image_size=image_size).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=0.0001)
optimizer_D = optim.Adam(D.parameters(), lr=0.0001)
optimizer_M = optim.Adam(M.parameters(), lr=0.0001)

#Schedulers
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.9)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.9)

loss_GAN = nn.BCELoss()
loss_MSE = nn.MSELoss()


Dlossarr =[]
Glossarr =[]
MSEarr =[]
ICarr = []

for epoch in trange(epochs+1):
    caps, real_imgs, masses = generate_sample(batch_size=16, image_size=image_size)
    caps, real_imgs, masses = caps.to(device), real_imgs.to(device), masses.to(device)

    caps_flat = caps.view(caps.size(0), -1)
    valid = torch.ones((caps.size(0), 1), device=device)
    fake = torch.zeros((caps.size(0), 1), device=device)

    # Generator step
    for _ in range(3):
        optimizer_G.zero_grad()
        caps_reshaped = caps.view(-1, 1, image_size, 1).expand(-1, 1, image_size, image_size)
        fake_imgs = G(caps_reshaped)
        cap_imgs = caps.view(caps.size(0), 1, image_size, 1).expand(-1, 1, image_size, image_size)
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
    #optimizer_M.zero_grad()
    #mass_pred = M(fake_imgs.detach())
    #m_loss = loss_MSE(mass_pred, masses)
    #m_loss.backward()
    #optimizer_M.step()

    # Mass loss
    fakeimgmass = MassCalculator(fake_imgs)
    realimgmass = MassCalculator(real_imgs)
    mloss = (abs(float(fakeimgmass) - float(realimgmass)) / float(realimgmass)) * 100
    with torch.no_grad():
        ic = imagecorrelation(cutoff(real_imgs[0].cpu().numpy(),0.5,tensorreturn=False), cutoff(fake_imgs[0].cpu().numpy(),0.5,tensorreturn=False))

    # Adjust learning rates
    adjust_learning_rate(optimizer_G, target_loss, g_loss)
    adjust_learning_rate(optimizer_D, target_loss, d_loss)



    if epoch % 20 == 0:
        Dlossarr.append(d_loss.item())
        Glossarr.append(g_loss.item())
        MSEarr.append(mloss)
        with torch.no_grad():
            ICarr.append(imagecorrelation(real_imgs[0].cpu().numpy(), fake_imgs[0].cpu().numpy()))

    # Visualization
    if epoch % 200 == 0:

        with torch.no_grad():
            print(
                f"[{epoch}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, Mass Error: {mloss:.2f}%, IC%: {ic:.2f}%")
            fig, axs = plt.subplots(2, 4, figsize=(10, 5))
            for i in range(4):
                axs[0, i].imshow(cutoff(real_imgs[i, 0].cpu(),0.5), cmap="viridis")
                axs[0, i].set_title("Real")
                axs[0, i].axis("off")
                axs[1, i].imshow(cutoff(fake_imgs[i, 0].cpu(),0.5), cmap="viridis")
                ic_value = imagecorrelation(
                    cutoff(real_imgs[i, 0].cpu().numpy(), 0.5, tensorreturn=False),
                    cutoff(fake_imgs[i, 0].cpu().numpy(), 0.5, tensorreturn=False)
                )
                mlossgraph = (abs(float(MassCalculator(fake_imgs[i,0])) - float(MassCalculator(real_imgs[i,0]))) / float(MassCalculator(real_imgs[i,0]))) * 100
                axs[1, i].set_title(f"IC: {ic_value:.2f}%, ME: {mlossgraph:.2f}%", fontsize=6)
                axs[1, i].axis("off")

            plt.tight_layout()
            #plt.show()

    #scheduler step
    scheduler_G.step()
    scheduler_D.step()

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