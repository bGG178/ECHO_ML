import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#https://www.sciencedirect.com/science/article/pii/S0955598624000463

#--HYPERPARAMS--
batch_size = 32

# U-Net style Generator with skip connections
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(UNetGenerator, self).__init__()
        # Encoder layers (Increases feature maps and reduces dimensions)
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

        # Bottleneck layer (Further reduces dimensions, abstract features)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )  # Output: (base_filters*4, H/8, W/8)

        # Decoder layers (Decreases feature maps and increases dimensions)
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

        # Generates output image
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

        # Resize capacitance_matrix
        capacitance_matrix_resized = F.interpolate(capacitance_matrix, size=generated_image.shape[2:], mode='bilinear',
                                                   align_corners=False)

        # Concatenate with the resized capacitance matrix
        concatenated_output = torch.cat([generated_image, capacitance_matrix_resized], dim=1)
        return concatenated_output

    def _crop_to_match(self, source, target):
        """
        Crops the source tensor to match the spatial dimensions of the target tensor. Probably not good practice but
        I couldn't get it working...
        """
        _, _, h, w = target.size()
        return source[:, :, :h, :w]


class Discriminator(nn.Module):
    def __init__(self, condition_shape=(1, 66, 66), in_channels=1):
        super(Discriminator, self).__init__()

        #Conv2D (3x3, stride 1) + MaxPooling2D (stride 2)
        self.fake_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #MaxPooling2D (stride 2)
        self.real_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Combined processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64 + 64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fully connected layers
        dummy_input = torch.zeros(1, 64 + 64, condition_shape[1] // 2, condition_shape[2] // 2)
        flattened_size = self._get_flattened_size(dummy_input)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1),
            nn.Sigmoid()
        )

    def _get_flattened_size(self, x):
        """
        Just for debugging purposes, to get the size of the flattened layer.
        :param x:
        :return:
        """
        x = self.conv_layers(x)
        return x.numel()

    def forward(self, image, condition):
        # Fake portion
        fake_features = self.fake_conv(condition)

        # Real portion
        real_features = self.real_pool(image)

        # Concatenate fake and real features
        combined_features = torch.cat([fake_features, real_features], dim=1)

        # Adjust the number of channels to match the expected input of conv_layers
        if combined_features.size(1) != 128:
            combined_features = nn.Conv2d(combined_features.size(1), 128, kernel_size=1)(combined_features)

        # Process through the rest of the discriminator
        x = self.conv_layers(combined_features)
        x = x.view(x.size(0), -1)  # Flatten
        validity = self.fc_layers(x)
        return validity


# Training loop for CGAN-ECT using U-Net generator
def train_cgan_ect(generator, discriminator, dataloader, num_epochs=3, device='cuda'):
    genlossarr = []
    dislossarr = []

    # Optimizers for both networks
    opt_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()  # Binary cross-entropy loss

    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for i, (real_imgs, cond_input) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            cond_input = cond_input.to(device)

            # Resize real images and condition input to a consistent size
            real_imgs = F.interpolate(real_imgs, size=(64, 64), mode='bilinear', align_corners=False)
            cond_input = F.interpolate(cond_input, size=(64, 64), mode='bilinear', align_corners=False)

            # Adversarial ground truths
            valid = torch.full((batch_size, 1), 0.9, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Generator (U-Net)
            # ---------------------
            opt_G.zero_grad()
            # Generator produces an image given the condition (raw capacitance) measurement.
            gen_imgs = generator(real_imgs,cond_input)
            # Resize generated images to match the discriminator's input size
            gen_imgs = F.interpolate(gen_imgs, size=(64, 64), mode='bilinear', align_corners=False)
            # Loss measures generator's ability to fool the discriminator (?)
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
                dislossarr.append(d_loss.item())
                genlossarr.append(g_loss.item())


    plt.plot(dislossarr, label='Discriminator Loss')
    plt.plot(genlossarr, label='Generator Loss')
    plt.show()
