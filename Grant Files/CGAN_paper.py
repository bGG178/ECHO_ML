import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#https://www.sciencedirect.com/science/article/pii/S0955598624000463

#--HYPERPARAMS--
noise_dim = 100  # Not used directly now, since U-Net generator uses only the condition input.
batch_size = 32


# U-Net style Generator with skip connections
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(UNetGenerator, self).__init__()
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )  # Output: (base_filters, H/2, W/2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )  # Output: (base_filters*2, H/4, W/4)

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )  # Output: (base_filters*4, H/8, W/8)

        # Decoder layers with skip connections
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )  # Output: (base_filters*2, H/4, W/4)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )  # Output: (base_filters, H/2, W/2)

        # Final layer to get desired output size
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  # Output: (out_channels, H, W)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # e.g., for input (1, 16, 15), output ~ (base_filters, 8, ~8)
        e2 = self.enc2(e1)  # (base_filters*2, 4, ~4)
        b = self.bottleneck(e2)  # (base_filters*4, 2, ~2)

        # Decoder with skip connection from encoder
        d2 = self.dec2(b)  # (base_filters*2, 4, ~4)
        d2 = torch.cat([d2, e2], dim=1)  # Concatenate along channel dim -> (base_filters*4, 4, ~4)

        d1 = self.dec1(d2)  # (base_filters, 8, ~8)
        d1 = torch.cat([d1, e1], dim=1)  # -> (base_filters*2, 8, ~8)

        out = self.final(d1)  # Upsample to (out_channels, ~16, ~16) or adjust depending on input dims
        return out


# Discriminator conditioned on the capacitance measurement
class Discriminator(nn.Module):
    def __init__(self, condition_shape=(1, 16, 15), in_channels=1):
        super(Discriminator, self).__init__()
        # The discriminator takes the generated (or real) image and the condition.
        # We upsample the condition to match the image spatial dimensions (assumed here to be 16x16 or similar).
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, kernel_size=4, stride=2, padding=1),  # image + condition
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),  # Adjust dimensions if image size changes
            nn.Sigmoid()  # Output probability
        )

    def forward(self, image, condition):
        # Upsample condition to match image size
        condition_upsampled = F.interpolate(condition, size=image.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate along the channel dimension
        x = torch.cat([image, condition_upsampled], dim=1)
        validity = self.disc(x)
        return validity


# Training loop for CGAN-ECT using U-Net generator
def train_cgan_ect(generator, discriminator, dataloader, num_epochs=100, device='cuda'):
    # Optimizers for both networks
    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()  # Binary cross-entropy loss

    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for i, (real_imgs, cond_input) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            cond_input = cond_input.to(device)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Generator (U-Net)
            # ---------------------
            opt_G.zero_grad()
            # Generator produces an image given the condition (raw capacitance) measurement.
            gen_imgs = generator(cond_input)
            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminator(gen_imgs, cond_input), valid)
            g_loss.backward()
            opt_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            # Loss for real images
            real_loss = criterion(discriminator(real_imgs, cond_input), valid)
            # Loss for fake images
            fake_loss = criterion(discriminator(gen_imgs.detach(), cond_input), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            if i % 50 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")


# Main code with dummy data
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create U-Net generator and discriminator instances
    generator = UNetGenerator(in_channels=1, out_channels=1, base_filters=64)
    discriminator = Discriminator(condition_shape=(1, 16, 15), in_channels=1)

    # Dummy dataset:
    # real_imgs: simulated tomography images (e.g., size 16x16, adjust as needed)
    # cond_input: raw capacitance measurements (16x15)
    real_images = torch.randn(100, 1, 16, 16)  # Real images (can be adjusted to the desired resolution)
    raw_capacitance = torch.randn(100, 1, 16, 15)  # 16x15 raw capacitance input
    dataset = TensorDataset(real_images, raw_capacitance)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the CGAN-ECT with U-Net generator
    train_cgan_ect(generator, discriminator, dataloader, num_epochs=10, device=device)
