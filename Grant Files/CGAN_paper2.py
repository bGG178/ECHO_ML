import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.types import Device
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from Construction.modulator import build_circulant_matrix, matrix_to_image
from Construction.phantom_generator import PhantomGenerator


"""
Make sure capacitances and phantoms go into training, but testing only on capacitances
"""

"""
Dropout method

self.conv_layers = nn.Sequential(
    nn.Conv2d(in_channels + 1, 64, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Dropout(0.3),  # Add dropout with a probability of 0.3
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Dropout(0.3),  # Add dropout
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Dropout(0.3),  # Add dropout
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
)
"""
"""
The train_cgan_ect function trains the Conditional GAN (CGAN) with:
The raw capacitance data (cond_input) as the condition input to the generator and discriminator.
The phantom data (real_imgs) as the target output for the generator and the real input for the discriminator.
The generator learns to map the capacitance data to the phantom data, while the discriminator ensures the generator produces realistic outputs.
"""


"""
Potential improvements

Here are some strategies to lower the generator loss in your GAN training:

### 1. **Adjust Learning Rates**
   - Reduce the learning rate of the generator or discriminator to stabilize training.
   - Example:
     ```python
     opt_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
     opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
     ```

### 2. **Balance Training Between Generator and Discriminator**
   - Train the generator more frequently than the discriminator (e.g., one generator step for every two discriminator steps).
   - Example:
     ```python
     if i % 2 == 0:  # Train generator every other step
         g_loss.backward()
         opt_G.step()
     ```

### 3. **Use Label Smoothing**
   - Apply label smoothing to the discriminator's labels to make it less confident, helping the generator learn better.
   - Example:
     ```python
     valid = torch.full((batch_size, 1), 0.9, device=device)  # Use 0.9 instead of 1.0
     fake = torch.zeros(batch_size, 1, device=device)
     ```

### 4. **Add Noise to Discriminator Inputs**
   - Add small random noise to the real and fake inputs to the discriminator to make it less confident.
   - Example:
     ```python
     real_imgs += torch.randn_like(real_imgs) * 0.05
     gen_imgs += torch.randn_like(gen_imgs) * 0.05
     ```

### 5. **Improve Generator Architecture**
   - Add more layers, use skip connections, or increase the number of filters in the generator to improve its capacity.

### 6. **Use a Different Loss Function**
   - Replace binary cross-entropy loss with a loss function like Wasserstein loss or Least Squares GAN loss for more stable training.
   - Example (Least Squares GAN):
     ```python
     criterion = nn.MSELoss()
     ```

### 7. **Pretrain the Generator**
   - Pretrain the generator on a simpler task (e.g., reconstructing real images) before adversarial training.

### 8. **Gradient Penalty**
   - Add a gradient penalty term to the discriminator loss to prevent it from becoming too strong (e.g., WGAN-GP).

### 9. **Normalize Inputs**
   - Ensure that all inputs (real images, generated images, and conditions) are properly normalized to the same range (e.g., [-1, 1]).

### 10. **Increase Batch Size**
   - Use a larger batch size to stabilize gradients and improve training dynamics.

By applying one or more of these strategies, you can help reduce the generator loss and improve the overall performance of your GAN.
"""

#https://www.sciencedirect.com/science/article/pii/S0955598624000463

#--HYPERPARAMS--
noise_dim = 100  # Not used directly now, since U-Net generator uses only the condition input.
batch_size = 32#32


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

    def forward(self, x, capacitance_matrix):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)

        # Decoder with skip connections
        d2 = self.dec2(b)
        d2 = torch.cat([d2, self._crop_to_match(d2, e2)], dim=1)
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, self._crop_to_match(d1, e1)], dim=1)

        # Final output
        generated_image = self.final(d1)

        # Ensure the output has 1 channel
        return generated_image[:, :1, :, :]  # Select only the first channel


    def _crop_to_match(self, source, target):
        """
        Crops the source tensor to match the spatial dimensions of the target tensor.
        """
        _, _, h, w = target.size()
        return source[:, :, :h, :w]


# Discriminator conditioned on the capacitance measurement
class Discriminator(nn.Module):
    def __init__(self, condition_shape=(1, 66, 66)):
        super(Discriminator, self).__init__()

        # 1) Image stream: only processes the phantom (1‑channel)
        self.image_conv = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.pool       = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2) Condition stream: only processes the capacitance (1‑channel)
        self.condition_conv = nn.Conv2d(1, 128, kernel_size=1)

        # 3) Joint conv stack: takes 256 channels (128+128)
        self.joint_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Compute flattened size dynamically
        dummy = torch.zeros(1, 256,
                            condition_shape[1] // 2,  # after one pool
                            condition_shape[2] // 2)
        flat_size = self._get_flattened_size(dummy)

        # 4) Final classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1),
            nn.Sigmoid()
        )

    def _get_flattened_size(self, x):
        x = self.joint_conv(x)
        return x.numel()

    def forward(self, real_image, fake_image, condition):
        # --- image path ---
        # real_image, fake_image: [B,1,H,W]
        r_feat = self.image_conv(real_image)
        r_feat = self.pool(r_feat)

        f_feat = self.image_conv(fake_image)
        f_feat = self.pool(f_feat)

        # --- condition path ---
        # condition: [B,1,H,W]
        c_feat = F.interpolate(condition, size=r_feat.shape[2:],
                               mode='bilinear', align_corners=False)
        c_feat = self.condition_conv(c_feat)
        # now c_feat is [B,128,H/2,W/2]

        # --- concatenate & joint conv ---
        real_cat = torch.cat([r_feat, c_feat], dim=1)  # [B,256,...]
        fake_cat = torch.cat([f_feat, c_feat], dim=1)

        real_out = self.fc(self.joint_conv(real_cat))
        fake_out = self.fc(self.joint_conv(fake_cat))

        return real_out, fake_out




# Training loop for CGAN-ECT using U-Net generator
def train_cgan_ect(generator, discriminator, dataloader, num_epochs=3, device='cuda'):
    genlossarr = []
    dislossarr = []

    # Optimizers for both networks
    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()  # Binary cross-entropy loss

    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for i, (real_imgs, cond_input) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)  # <- This is the phantom
            cond_input = cond_input.to(device)  # <- This is the capacitance measurement

            # Resize both to same size
            real_imgs = F.interpolate(real_imgs, size=(64, 64), mode='bilinear', align_corners=False)
            cond_input = F.interpolate(cond_input, size=(64, 64), mode='bilinear', align_corners=False)

            valid = torch.full((batch_size, 1), 0.9, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # --------------------
            #  Train Generator
            # --------------------
            opt_G.zero_grad()
            gen_imgs = generator(cond_input, cond_input)  # generator(input_image, capacitance)
            gen_imgs = F.interpolate(gen_imgs, size=(64, 64), mode='bilinear', align_corners=False)

            g_loss = criterion(discriminator(gen_imgs, gen_imgs, cond_input)[1], valid)
            g_loss.backward()
            opt_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs, real_imgs, cond_input)[0], valid)
            fake_loss = criterion(discriminator(real_imgs, gen_imgs.detach(), cond_input)[1], fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            if i % 50 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                dislossarr.append(d_loss.item())
                genlossarr.append(g_loss.item())


    plt.plot(dislossarr, label='Discriminator Loss')
    plt.plot(genlossarr, label='Generator Loss')
    plt.show()


# Main code with dummy data
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # Create U-Net generator and discriminator instances
    generator = UNetGenerator(in_channels=1, out_channels=1, base_filters=64)
    discriminator = Discriminator(condition_shape=(1, 66, 66))

    # Load the combined dataset
    datafile = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData\combined_data.npy"
    combined_data = np.load(datafile, allow_pickle=True)

    # Split the dataset into training (90%) and testing (10%)
    train_size = int(0.9 * len(combined_data))
    train_data = combined_data[:train_size]
    test_data = combined_data[train_size:]


    """WRONG, for some reason its piping measurements into both capacitance and real img, when it should just be doing
    capacitance. Fix!"""
    # Prepare the training dataset
    phantom_generator = PhantomGenerator(12, 1, 128)

    train_real_images = torch.tensor(
        np.array([phantom_generator.generate_phantom(data['objects']) for data in train_data]),
        dtype=torch.float32
    ).unsqueeze(1)
    train_real_images = (train_real_images - 0.5) * 2  # Normalize to [-1, 1]

    train_capacitance = torch.tensor(
        np.array([build_circulant_matrix(data['measurements']) for data in train_data]),
        dtype=torch.float32
    ).unsqueeze(1)
    train_capacitance = (train_capacitance - 0.5) * 2  # Normalize to [-1, 1]

    train_dataset = TensorDataset(train_real_images, train_capacitance)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # Prepare the testing dataset
    test_real_images = torch.tensor(
        np.array([phantom_generator.generate_phantom(data['objects']) for data in test_data]),
        dtype=torch.float32
    ).unsqueeze(1)
    test_real_images = (test_real_images - 0.5) * 2  # Normalize to [-1, 1]

    test_capacitance = torch.tensor(
        np.array([build_circulant_matrix(data['measurements']) for data in test_data]),
        dtype=torch.float32
    ).unsqueeze(1)
    test_capacitance = (test_capacitance - 0.5) * 2  # Normalize to [-1, 1]

    test_dataset = TensorDataset(test_real_images, test_capacitance)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #for i, (real_img, cond_input) in enumerate(test_dataloader):
    #    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    #    ax1.set_title("Actual Phantom")
    #    ax1.imshow(real_img.cpu().squeeze(0).squeeze(0), cmap='gray')
    #    ax2.imshow(cond_input.cpu().squeeze(0).squeeze(0), cmap='gray')
    #    plt.tight_layout()
    #    plt.show()

    # TRAINING!!!Train the CGAN-ECT with U-Net generator!!!!!!!!!!!!!!!!!
    train_cgan_ect(generator, discriminator, train_dataloader, num_epochs=1, device=device)

    # Testing loop with visualization
    generator.eval()
    with torch.no_grad():
        for i, (real_img, cond_input) in enumerate(test_dataloader):
            real_img = real_img.to(device)
            cond_input = cond_input.to(device)

            reconstructed_img = generator(cond_input, cond_input).to(device).cpu()

            # Print the shape of the generated image
            #print(f"Shape of reconstructed_img after generation: {reconstructed_img.shape}")
            # Expected: (1, 2, 64, 64) based on the error

            # Ensure the image is 2D by selecting the first channel or averaging across channels
            reconstructed_img = reconstructed_img[0].squeeze(0)  # Select the first channel
            #print(f"Shape of reconstructed_img after squeezing: {reconstructed_img.shape}")
            # Expected: (2, 64, 64)

            reconstructed_img = reconstructed_img[0]
            # reconstructed_img = reconstructed_img.mean(dim=0)
            #print(f"Shape of reconstructed_img after selecting the first channel: {reconstructed_img.shape}")
            # Expected: (64, 64)

            # Alternatively, use the mean across channels:
            # reconstructed_img = reconstructed_img.mean(dim=0)
            # print(f"Shape of reconstructed_img after averaging channels: {reconstructed_img.shape}")
            # Expected: (64, 64)

            # Normalize the image for display
            #print(
            #    f"Min and Max of reconstructed_img before normalization: {reconstructed_img.min()}, {reconstructed_img.max()}")
            reconstructed_img -= reconstructed_img.min()
            reconstructed_img /= reconstructed_img.max()
           # print(
            #    f"Min and Max of reconstructed_img after normalization: {reconstructed_img.min()}, {reconstructed_img.max()}")
            # Expected: Min = 0.0, Max = 1.0

            # Display every 100th test result
            if i % 100 == 0:
                real_img = real_img.cpu().squeeze(0).squeeze(0)
                cond_input = cond_input.cpu().squeeze(0).squeeze(0)

                # Plot the actual phantom, capacitance image, and reconstruction
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(real_img, cmap='gray')
                ax1.set_title("Actual Phantom")
                ax2.imshow(cond_input, cmap='gray')
                ax2.set_title("Capacitance Image")
                ax3.imshow(reconstructed_img, cmap='gray')
                ax3.set_title("Reconstruction Attempt")
                plt.tight_layout()
                plt.show()

    # Dummy dataset:
    # real_imgs: simulated tomography images (e.g., size 16x16, adjust as needed)
    # cond_input: raw capacitance measurements (16x15)
    #real_images = torch.randn(100, 1, 16, 16)  # Real images (can be adjusted to the desired resolution)
    #raw_capacitance = torch.randn(100, 1, 16, 15)  # 16x15 raw capacitance input
    #dataset = TensorDataset(real_images, raw_capacitance)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the CGAN-ECT with U-Net generator
    #train_cgan_ect(generator, discriminator, dataloader, num_epochs=10, device=device)