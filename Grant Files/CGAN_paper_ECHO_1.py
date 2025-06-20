import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
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
    """
    This class defines a U-Net like Generator neural network.
    U-Net architectures are characterized by an encoder-decoder structure with
    skip connections. The encoder progressively downsamples the input, capturing
    context, while the decoder progressively upsamples it, localizing features.
    Skip connections directly link layers from the encoder to corresponding
    layers in the decoder, helping to preserve high-resolution information.
    """
    def __init__(self, image_size=128, base_filters=64):
        """
        Constructor for the Generator class.
        Initializes all the layers of the U-Net architecture.

        Args:
            image_size (int, optional): The expected height and width of the output image.
                                        Defaults to 128. This is used in the forward pass
                                        to reshape the final output.
            base_filters (int, optional): The number of filters in the first convolutional
                                          layer of the encoder (and the last of the decoder).
                                          The number of filters in subsequent layers is typically
                                          a multiple of this value. Defaults to 64.
        """
        super().__init__() # Call the constructor of the parent class (nn.Module)
                           # This is essential for correctly setting up the nn.Module.

        # --- Encoder Path (Downsampling) ---
        # The encoder part of the U-Net progressively reduces the spatial dimensions
        # of the input while increasing the number of feature channels.

        # Encoder Block 1 (enc1)
        self.enc1 = nn.Sequential(
            # nn.Conv2d: 2D Convolutional Layer
            #   1: in_channels - Number of channels in the input image.
            #      For a grayscale image, this is 1. For an RGB image, it would be 3.
            #      Here, it's 1, suggesting the input is a single-channel image (e.g., grayscale).
            #   base_filters: out_channels - Number of filters (feature maps) to produce.
            #                   This will be the number of channels in the output of this layer.
            #   kernel_size=4: The size of the convolutional kernel (4x4). (slides across image)
            #   stride=2: The step size of the convolution. A stride of 2 halves the
            #             spatial dimensions (height and width) of the input. A stride of 2 moves the kernel by 2 instead of 1
            #   padding=1: The amount of zero-padding added to the sides of the input.
            #              Padding helps control the output dimensions. With kernel_size=4,
            #              stride=2, padding=1, the output size is H/2 x W/2.
            nn.Conv2d(1, base_filters, kernel_size=4, stride=2, padding=1),

            # nn.BatchNorm2d: Batch Normalization for 2D inputs.
            #   base_filters: num_features - The number of feature channels from the
            #                   preceding convolutional layer. Batch normalization helps
            #                   stabilize and accelerate training by normalizing the
            #                   activations of the current batch.
            nn.BatchNorm2d(base_filters),

            # nn.ReLU: Rectified Linear Unit activation function.
            #   inplace=True: If True, performs the operation in-place, modifying the
            #                 input tensor directly to save memory. This means the
            #                 output tensor will be the same object as the input tensor.
            #                 It can be slightly more memory efficient but requires care
            #                 if the input tensor is needed elsewhere unmodified.
            nn.ReLU(inplace=True)
        )

        # Encoder Block 2 (enc2)
        self.enc2 = nn.Sequential(
            # The input to this block is the output of enc1, which has 'base_filters' channels.
            #   base_filters: in_channels
            #   base_filters * 2: out_channels - Doubling the number of filters.
            #   kernel_size=4, stride=2, padding=1: Again, downsampling by a factor of 2.
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        # --- Bottleneck ---
        # The bottleneck is the deepest part of the U-Net, where the spatial
        # dimensions are smallest, and the feature representation is most compressed.

        self.bottleneck = nn.Sequential(
            # Input has 'base_filters * 2' channels from enc2.
            #   base_filters * 2: in_channels
            #   base_filters * 4: out_channels - Doubling filters again.
            #   kernel_size=4, stride=2, padding=1: Further downsampling.
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        # --- Decoder Path (Upsampling) ---
        # The decoder part of the U-Net progressively increases the spatial dimensions
        # (upsampling) while reducing the number of feature channels. It also incorporates
        # information from the encoder via skip connections.

        # Decoder Block 2 (dec2) - Corresponds to enc2
        self.dec2 = nn.Sequential(
            # nn.Upsample: Upsamples the input.
            #   scale_factor=2: Multiplies the input's height and width by 2.
            #   mode='bilinear': The upsampling algorithm. Bilinear interpolation is
            #                    a common choice.
            #   align_corners=False: This argument relates to how grid points are aligned
            #                        during interpolation. Setting it to False is often
            #                        recommended as it's more consistent with how frameworks
            #                        like OpenCV handle image resizing and is the default in
            #                        newer PyTorch versions for some modes.
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # nn.Conv2d: A regular convolution, not a transposed convolution.
            #            Here it's used after upsampling to refine features.
            #   base_filters * 4: in_channels - From the bottleneck's output.
            #                     Note: In a typical U-Net, after concatenation with a skip
            #                     connection, the input channels to the conv layer would be
            #                     (bottleneck_output_channels + corresponding_encoder_output_channels).
            #                     However, this Conv2d happens *before* the skip connection
            #                     is concatenated in the forward pass. This is a valid design choice.
            #                     The skip connection from enc2 will be added later in the forward pass.
            #   base_filters * 2: out_channels - Halving the number of filters from the bottleneck.
            #   kernel_size=3, padding=1: These parameters typically maintain the spatial
            #                             resolution (if stride=1, which is default).
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder Block 1 (dec1) - Corresponds to enc1
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.Conv2d:
            #   base_filters * 2 + base_filters*2 : in_channels
            #     - The first 'base_filters * 2' comes from the output of the previous
            #       decoder block (dec2, after its Conv2d and ReLU).
            #     - The second 'base_filters * 2' comes from the skip connection from
            #       the corresponding encoder block (enc2). This concatenation happens
            #       in the `forward` method before this layer is applied.
            #       So, this Conv2d processes the concatenated tensor.
            #   base_filters: out_channels - Further reducing filter count.
            #   kernel_size=3, padding=1: Maintaining spatial resolution.
            nn.Conv2d(base_filters * 2 + base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # --- Final Output Layer ---
        # This layer produces the final output, typically an image.

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.Conv2d:
            #   base_filters + base_filters: in_channels
            #     - The first 'base_filters' comes from the output of dec1.
            #     - The second 'base_filters' comes from the skip connection from enc1.
            #       This concatenation also happens in the `forward` method.
            #   1: out_channels - This means the output will be a single-channel image
            #                    (e.g., grayscale). If an RGB image was desired, this would be 3.
            #   kernel_size=3, padding=1: Maintaining spatial resolution.
            nn.Conv2d(base_filters + base_filters, 1, kernel_size=3, padding=1),

            # nn.Tanh: Tanh activation function.
            #          This squashes the output values to the range [-1, 1].
            #          This is common for image generation if the input images
            #          were also normalized to this range.
            nn.Tanh()
        )

        # Store the target image size for reshaping in the forward pass.
        self.image_size = image_size

    def forward(self, x):
        """
        Defines the forward pass of the Generator.
        This method specifies how the input 'x' flows through the network layers.

        Args:
            x (torch.Tensor): The input tensor to the generator.
                              Expected shape for a single-channel image input might be
                              (batch_size, 1, height, width), where height and width
                              are appropriately sized for the first Conv2d layer.

        Returns:
            torch.Tensor: The generated image tensor.
                          Its shape will be (batch_size, 1, self.image_size, self.image_size).
        """
        # --- Encoder Path ---
        # Pass the input 'x' through the first encoder block.
        e1 = self.enc1(x)   # Output of enc1, to be used in a skip connection later.
                            # Shape example: if x is (N, 1, 128, 128), e1 is (N, base_filters, 64, 64)

        # Pass the output of enc1 through the second encoder block.
        e2 = self.enc2(e1)  # Output of enc2, also for a skip connection.
                            # Shape example: e2 is (N, base_filters*2, 32, 32)

        # Pass the output of enc2 through the bottleneck.
        b = self.bottleneck(e2) # Output of the bottleneck.
                                # Shape example: b is (N, base_filters*4, 16, 16)

        # --- Decoder Path with Skip Connections ---
        # Pass the bottleneck output through the first decoder block (dec2 in our naming).
        # Note: The nn.Sequential for self.dec2 already includes Upsample and Conv2D.
        d2_upsampled_conved = self.dec2(b) # Output after upsampling and convolution within dec2's Sequential.
                                # Shape example: d2_upsampled_conved is (N, base_filters*2, 32, 32)

        # Skip Connection: Concatenate the output of d2_upsampled_conved with the output of enc2 (e2).
        # torch.cat concatenates tensors along a specified dimension.
        #   [d2_upsampled_conved, e2]: A list of tensors to concatenate.
        #   dim=1: Concatenate along the channel dimension (dimension 1).
        #          Channels are typically dimension 1 in PyTorch (N, C, H, W).
        #          So, if d2_upsampled_conved has C1 channels and e2 has C2 channels,
        #          the result will have C1 + C2 channels.
        #          Here, d2_upsampled_conved has base_filters*2 channels, and e2 also has base_filters*2 channels.
        #          The concatenated tensor will have base_filters*2 + base_filters*2 channels.
        d2_concatenated = torch.cat([d2_upsampled_conved, e2], dim=1)
        # Shape example: d2_concatenated is (N, base_filters*2 + base_filters*2, 32, 32)

        # Pass the concatenated tensor through the next decoder block (dec1 in our naming).
        # The nn.Sequential for self.dec1 expects an input with (base_filters*2 + base_filters*2) channels
        # for its Conv2d layer, which matches d2_concatenated.
        d1_upsampled_conved = self.dec1(d2_concatenated) # Output after upsampling and convolution within dec1's Sequential.
                                    # Shape example: d1_upsampled_conved is (N, base_filters, 64, 64)

        # Skip Connection: Concatenate the output of d1_upsampled_conved with the output of enc1 (e1).
        d1_concatenated = torch.cat([d1_upsampled_conved, e1], dim=1)
        # Shape example: d1_concatenated is (N, base_filters + base_filters, 64, 64)

        # Pass the concatenated tensor through the final output layer.
        # The nn.Sequential for self.final expects an input with (base_filters + base_filters) channels.
        generated_image = self.final(d1_concatenated)
        # Shape example: generated_image is (N, 1, 128, 128) after upsampling and final conv.

        # Reshape the output tensor to ensure it matches the desired image format.
        #   .view() is a PyTorch method for reshaping tensors.
        #   -1: Infers the batch size (the first dimension).
        #   1: Number of channels (as defined in the final Conv2d layer).
        #   self.image_size: Height of the output image.
        #   self.image_size: Width of the output image.
        # This step is a safeguard or explicit reshaping, though if the layers are
        # designed correctly, the output of self.final might already be in this shape.
        return generated_image.view(-1, 1, self.image_size, self.image_size)

class Discriminator(nn.Module):
    """
    This class defines a Discriminator neural network.
    In a GAN setup, the Discriminator's role is to classify its input
    as either 'real' or 'fake'. It learns by being shown examples of
    actual data and data produced by a Generator network.

    This specific Discriminator seems to take two images (a real one and a fake one),
    concatenates them, and then processes them through a series of convolutional
    and linear layers to output a single probability score.
    """
    def __init__(self, image_size=128):
        """
        Constructor for the Discriminator class.
        Initializes all the layers of the network.

        Args:
            image_size (int, optional): The height and width of the input images.
                                        Defaults to 128. This is used to calculate
                                        the input size for the fully connected layer (nn.Linear)
                                        after the convolutional feature extraction.
        """
        super().__init__() # Call the constructor of the parent class (nn.Module)
                           # This is essential for correctly setting up the nn.Module.

        # self.model defines the architecture of the discriminator as a sequence of layers.
        # nn.Sequential is a container that passes data through a sequence of modules
        # in the order they are defined.
        self.model = nn.Sequential(
            # --- First Convolutional Block ---
            # nn.Conv2d: 2D Convolutional Layer for feature extraction.
            #   2: in_channels - Number of channels in the input tensor.
            #      This is '2' because in the forward pass, a real image (1 channel, assuming
            #      grayscale) and a fake image (1 channel) will be concatenated along the
            #      channel dimension, resulting in a 2-channel input to this layer.
            #      If the input images were RGB (3 channels each), and they were concatenated,
            #      this would be 6.
            #   64: out_channels - Number of filters (feature maps) to produce.
            #                    This layer will output 64 feature channels.
            #   kernel_size=4: The size of the convolutional kernel (4x4).
            #   stride=2: The step size of the convolution. A stride of 2 halves the
            #             spatial dimensions (height and width) of the input.
            #   padding=1: The amount of zero-padding added to the sides of the input.
            #              With kernel_size=4, stride=2, padding=1, the output size is H/2 x W/2.
            #              So, if input is 128x128, output of this conv layer is 64x64.
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),

            # nn.LeakyReLU: Leaky Rectified Linear Unit activation function.
            #   0.2: negative_slope - Controls the angle of the negative slope.
            #        Unlike standard ReLU (which outputs 0 for negative inputs),
            #        Leaky ReLU allows a small, non-zero gradient for negative inputs
            #        (output = negative_slope * input if input < 0). This can help
            #        prevent "dying ReLU" problems and is common in GANs.
            #   inplace=True: If True, performs the operation in-place, modifying the
            #                 input tensor directly to save memory.
            nn.LeakyReLU(0.2, inplace=True),

            # --- Second Convolutional Block ---
            # nn.Conv2d: Another 2D Convolutional Layer.
            #   64: in_channels - Must match the out_channels of the previous Conv2d layer.
            #   128: out_channels - Number of filters, increasing the feature depth.
            #   kernel_size=4, stride=2, padding=1: Same kernel parameters, further
            #                                        halving the spatial dimensions.
            #              If input to this layer is 64x64, output is 32x32.
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),

            # nn.BatchNorm2d: Batch Normalization for 2D inputs.
            #   128: num_features - The number of feature channels from the
            #                    preceding convolutional layer. Batch normalization
            #                    helps stabilize and accelerate training. It's applied
            #                    before the activation function in this block.
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True), # Activation after batch normalization.

            # --- Flattening Layer ---
            # nn.Flatten: Flattens a contiguous range of dimensions into a tensor.
            #             By default, it flattens all dimensions starting from dim 1 (the first
            #             dimension after the batch dimension).
            #             The output of the previous LeakyReLU is a 4D tensor of shape
            #             (batch_size, 128, image_size/4, image_size/4).
            #             For example, if image_size is 128, the shape is (batch_size, 128, 32, 32).
            #             Flattening this will result in a 2D tensor of shape
            #             (batch_size, 128 * 32 * 32).
            nn.Flatten(),

            # --- Fully Connected (Linear) Layer ---
            # nn.Linear: Applies a linear transformation to the incoming data (y = xA^T + b).
            #            This is a standard fully connected layer.
            #   in_features: The number of input features. This must match the number of
            #                elements in the flattened output from the previous layer.
            #                The input spatial dimension to this layer is image_size divided by 2 (from first conv)
            #                and then divided by 2 again (from second conv), so image_size // 4.
            #                So, the number of input features is 128 (channels) * (image_size//4) * (image_size//4).
            #                Example: if image_size = 128, then image_size//4 = 32.
            #                         in_features = 128 * 32 * 32 = 128 * 1024 = 131072.
            #   out_features: 1 - The number of output features. Since this is a binary
            #                 classification task (real vs. fake), a single output neuron
            #                 is used to produce a raw score (logit).
            nn.Linear(128 * (image_size//4) * (image_size//4), 1),

            # --- Output Activation ---
            # nn.Sigmoid: Sigmoid activation function.
            #             Squashes the output of the linear layer to a range between 0 and 1.
            #             This output can be interpreted as the probability that the input
            #             is 'real' (or 'fake', depending on how the loss function is set up).
            #             A value close to 1 typically means 'real', and close to 0 means 'fake'.
            nn.Sigmoid()
        )

    def forward(self, real_imgs, fake_imgs):
        """
        Defines the forward pass of the Discriminator.

        Args:
            real_imgs (torch.Tensor): A batch of real images.
                                      Expected shape: (batch_size, 1, image_size, image_size)
                                      assuming single-channel (e.g., grayscale) images.
            fake_imgs (torch.Tensor): A batch of fake images generated by the Generator.
                                      Expected shape: (batch_size, 1, image_size, image_size)
                                      assuming single-channel images.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing the probability
                          scores from the Sigmoid activation, indicating the likelihood
                          that the *concatenated pair* (or implicitly, the fake image
                          in relation to the real one if designed for a conditional setup)
                          is classified as real.
        """
        # Concatenate real and fake images along the channel dimension (dim=1).
        # If real_imgs is (N, 1, H, W) and fake_imgs is (N, 1, H, W),
        # then x will be (N, 1+1, H, W) = (N, 2, H, W).
        # This means the network processes the real and fake images "side-by-side"
        # in terms of their channels. The first Conv2d layer is configured with
        # in_channels=2 to handle this concatenated input.
        # This approach can be used in various GAN setups, for example:
        # 1. In some conditional GANs (like Pix2Pix), the input image (condition)
        #    and the generated/real target image are concatenated and fed to the discriminator.
        #    Here, it seems 'real_imgs' might act as a condition and 'fake_imgs' as the
        #    image to be judged against that condition, or vice-versa, or they are simply
        #    paired for joint evaluation.
        # 2. It could be a way to provide more context to the discriminator by allowing it
        #    to directly compare features from both simultaneously.
        x = torch.cat([real_imgs, fake_imgs], dim=1)

        # Pass the concatenated tensor through the sequential model defined in __init__.
        # The model will perform convolutions, activations, batch normalization,
        # flattening, and finally a linear transformation followed by a sigmoid
        # to output a single probability score per input pair in the batch.
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
    datafile = r"C:\Users\bower\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData\npg_3500_15e15_traintest.json"


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
        for i, (real_imgs, cond_input) in enumerate((train_dataloader)):

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
    output_file = os.path.join(modeloutput, f"{electrode_count}e{emitting_count}_ECHO_0.2_model.pth")
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

