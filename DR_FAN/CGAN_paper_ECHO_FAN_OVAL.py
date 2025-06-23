import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.path import Path
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast,GradScaler
import os
import json
import torch.nn.functional as F
from shapely.geometry import Point
import numpy as np
import modulator_FAN_OVAL as m2
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
        shape_type = obj["type"]
        xf = obj.get("xf", 1.0)
        yf = obj.get("yf", 1.0)
        rot = obj.get("rotation", 0.0)  # degrees
        theta = np.radians(rot)

        if shape_type == "circle":
            mask = (xv - cx) ** 2 + (yv - cy) ** 2 <= r ** 2
            img[mask] = 1.0

        elif shape_type == "oval":
            # Rotate coordinates back by -theta
            x_shifted = xv - cx
            y_shifted = yv - cy
            x_rot = np.cos(-theta) * x_shifted + np.sin(-theta) * y_shifted
            y_rot = -np.sin(-theta) * x_shifted + np.cos(-theta) * y_shifted

            # Ellipse equation: (x/xf*r)^2 + (y/yf*r)^2 <= 1
            mask = (x_rot / (xf * r)) ** 2 + (y_rot / (yf * r)) ** 2 <= 1
            img[mask] = 1.0

        elif shape_type == "square":
            # Define square corners centered at origin
            half = r
            corners = np.array([
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half]
            ])

            # Rotate and shift square corners
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_corners = corners @ rotation_matrix.T + np.array([cx, cy])

            # Create path from rotated square
            square_path = Path(rotated_corners)

            # Flatten grid points and check which are inside polygon
            points = np.column_stack((xv.ravel(), yv.ravel()))
            mask = square_path.contains_points(points).reshape((image_size, image_size))
            img[mask] = 1.0

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

def create_circular_mask(h, w, device='cpu', tensor_type='torch'):
    """
    Creates a circular mask for an image of size h x w.
    The circle is inscribed in the rectangle h x w.
    For a square image (h=w=N), radius is N/2, center is (N-1)/2.
    """
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    # For an inscribed circle in a square, radius is min(h,w)/2
    # If it's always a square matrix as input, h=w.
    radius = min(h, w) / 2.0

    if tensor_type == 'torch':
        Y, X = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        dist_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
    elif tensor_type == 'numpy':
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
    else:
        raise ValueError("tensor_type must be 'torch' or 'numpy'")
    return mask

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
    def __init__(self, image_size=128, base_filters=96):
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
            #nn.Sigmoid()
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


def MassCalculator_circular(img_input):
    """
    Calculates the mass of the image within an inscribed circle
    by summing pixel values in that circle.
    Input can be a PyTorch tensor or a NumPy array (assumed 2D or more with H,W as last two dims).
    Returns a Python float.
    """
    is_torch_tensor = isinstance(img_input, torch.Tensor)

    if is_torch_tensor:
        img_tensor = img_input
        h, w = img_tensor.shape[-2], img_tensor.shape[-1]  # Assuming H, W are last two dimensions
        mask = create_circular_mask(h, w, device=img_tensor.device, tensor_type='torch')
    elif isinstance(img_input, np.ndarray):
        img_array = img_input
        h, w = img_array.shape[-2], img_array.shape[-1]
        mask = create_circular_mask(h, w, tensor_type='numpy')
        img_tensor = torch.from_numpy(img_array)  # Convert to tensor for consistency in sum
    else:
        raise TypeError("Input to MassCalculator_circular must be a PyTorch tensor or NumPy array.")

    # Ensure mask is broadcastable if img_tensor has more than 2 dims (e.g., batch or channels)
    # Example: if img_tensor is (C,H,W), mask (H,W) will apply fine.
    # If img_tensor is (N,C,H,W), mask (H,W) will also apply fine.

    # Apply mask
    masked_img = img_tensor * mask.unsqueeze(0).unsqueeze(0) if img_tensor.ndim == 4 else \
        img_tensor * mask.unsqueeze(0) if img_tensor.ndim == 3 else \
            img_tensor * mask
    # Or, more robustly for selecting elements:
    # For summing, we can just select the elements.
    # Flatten the image and mask for selection if dealing with multiple dimensions easily.
    # Assuming we sum over all dimensions except batch if present.

    # Simpler: select pixels within the mask and sum them
    if is_torch_tensor:
        # For PyTorch tensors, ensure mask is boolean for indexing
        circular_pixels = img_tensor[..., mask]  # This flattens the selected pixels
    else:  # Was NumPy, converted to img_tensor
        circular_pixels = img_tensor[..., mask]  # Mask was NumPy boolean, works here too

    total_mass_in_circle = torch.sum(circular_pixels.float())
    return total_mass_in_circle.item()


# If you need the differentiable version:
def MassCalculator_circular_differentiable(img_tensor: torch.Tensor):
    """
    Calculates the mass of the image (PyTorch tensor) within an inscribed circle.
    Returns a PyTorch scalar tensor.
    """
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    mask = create_circular_mask(h, w, device=img_tensor.device, tensor_type='torch')

    # Select pixels within the mask and sum them
    # For an image (H,W) or (C,H,W) or (N,C,H,W)
    # We need to ensure mask is applied correctly to H,W dimensions

    # If img_tensor is (H,W), circular_pixels = img_tensor[mask]
    # If img_tensor is (C,H,W), circular_pixels = img_tensor[:, mask] and then sum over all.
    # If img_tensor is (N,C,H,W), circular_pixels = img_tensor[:, :, mask] and then sum over all.

    # A robust way is to multiply by mask and then sum.
    # Ensure mask is broadcastable. Mask is (H,W).
    # If img_tensor is (C,H,W), mask needs to be (1,H,W) for broadcasting.
    # If img_tensor is (N,C,H,W), mask needs to be (1,1,H,W) for broadcasting.

    # Reshape mask for broadcasting:
    # Get number of dimensions to unsqueeze for mask
    num_unsqueeze_dims = img_tensor.ndim - 2
    broadcastable_mask = mask
    for _ in range(num_unsqueeze_dims):
        broadcastable_mask = broadcastable_mask.unsqueeze(0)

    masked_img_values = img_tensor * broadcastable_mask
    total_mass_in_circle = torch.sum(masked_img_values.float())  # Sum all elements of the product

    return total_mass_in_circle


def imagecorrelation_circular(realarr_input, fakearr_input):
    """
    Calculates Pearson correlation between two input arrays (e.g., images)
    considering only pixels within an inscribed circle.
    Inputs are assumed to be NumPy arrays, potentially 2D (H,W) or
    3D with a leading singleton channel dimension (1,H,W).
    Returns a score between 0 and 100.
    """
    if not (isinstance(realarr_input, np.ndarray) and isinstance(fakearr_input, np.ndarray)):
        raise TypeError("Inputs must be NumPy ndarrays.")

    # --- Modification to handle (1,H,W) inputs ---
    processed_realarr = realarr_input
    if realarr_input.ndim == 3 and realarr_input.shape[0] == 1:
        processed_realarr = realarr_input.squeeze(axis=0)
    elif realarr_input.ndim != 2:
        raise ValueError(
            f"Input realarr must be 2D or 3D with a leading singleton dimension. Got shape: {realarr_input.shape}")

    processed_fakearr = fakearr_input
    if fakearr_input.ndim == 3 and fakearr_input.shape[0] == 1:
        processed_fakearr = fakearr_input.squeeze(axis=0)
    elif fakearr_input.ndim != 2:
        raise ValueError(
            f"Input fakearr must be 2D or 3D with a leading singleton dimension. Got shape: {fakearr_input.shape}")
    # --- End of modification ---

    # Now, processed_realarr and processed_fakearr should be 2D
    if processed_realarr.shape != processed_fakearr.shape:
        raise ValueError(
            f"Input arrays must have the same HxW shape after processing. Got {processed_realarr.shape} and {processed_fakearr.shape}")

    # Assuming square inputs for a 'perfectly' inscribed circle as per original intent.
    # if processed_realarr.shape[0] != processed_realarr.shape[1]:
    #     print("Warning: imagecorrelation_circular expects square arrays for a perfectly inscribed circle.")

    h, w = processed_realarr.shape
    mask = create_circular_mask(h, w, tensor_type='numpy')  # Get NumPy boolean mask

    real_pixels_in_circle = processed_realarr[mask]  # This flattens the selection
    fake_pixels_in_circle = processed_fakearr[mask]

    if real_pixels_in_circle.size < 2:  # Need at least 2 points for correlation
        # This can happen if the mask is too small or image dimensions are tiny
        return 0.0

    std_real = np.std(real_pixels_in_circle)
    std_fake = np.std(fake_pixels_in_circle)

    # If both arrays are constant AND identical within the circle, corr is 1.
    # If one is constant, or both are different constants, corr is undefined/0.
    if std_real < 1e-9 and std_fake < 1e-9:  # Both are constant within the circle
        return 100.0 if np.array_equal(real_pixels_in_circle, fake_pixels_in_circle) else 0.0
    if std_real < 1e-9 or std_fake < 1e-9:  # Only one is constant
        return 0.0

    correlation = np.corrcoef(real_pixels_in_circle, fake_pixels_in_circle)[0, 1]

    if np.isnan(correlation):  # Should be less likely with std dev checks, but good to have
        # This could happen if somehow one array ended up with all same values after masking
        # and the std check was borderline.
        # If they are truly identical, correlation should be 1.
        if np.array_equal(real_pixels_in_circle, fake_pixels_in_circle):
            return 100.0
        return 0.0

    return max(0, correlation * 100)

"""
Old Image Correlation Function
def imagecorrelation(realarr, fakearr):

realarr = realarr.flatten()

fakearr = fakearr.flatten()

realarr = (realarr - np.mean(realarr)) / (np.std(realarr) + 1e-8)

fakearr = (fakearr - np.mean(fakearr)) / (np.std(fakearr) + 1e-8)

correlation = np.corrcoef(realarr, fakearr)[0, 1]

return max(0, correlation * 100) # keep it in 0–100 range
"""

"""
Old Mass Error Function
def MassCalculator(img):
Calculates the mass of the image by summing all pixel values.

return float(torch.sum(torch.tensor(img)))
"""

if __name__ == "__main__":
    #######################################################HYPERPARAMETERS!!!!#############################################################################################
    batch_size = 16
    image_size = 64 #WORKS WELL FOR 32x32 and 64x64, 128x128 needs work!
    target_loss = 0.45
    epochs = 5
    scheduler_step_size = 1000  # Step size for the learning rate scheduler
                            # Try raising this to 200, was 100
    Lambda_L1_value = 120 # Weight for the L1 loss in the generator, was 150
                            # Try raising this to 150, try lowering to 75
    electrode_count = 15
    emitting_count = 15
    datafile = "Data/npgC_10000_15e15_traintest.json"
    overviewsavelocation = "Outputs/Oval/TrainingOverview__npgOval20000.png"


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

  #  if hasattr(torch, 'compile'):
        # Compile the models if torch.compile is available
 #       G = torch.compile(G)
 #       D = torch.compile(D)
 #       print("Compiling...")

    #Initialize the optimization functions
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0001)

    #Initialize the scheduler functions
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=scheduler_step_size, gamma=0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=scheduler_step_size, gamma=0.9)

    #Initialize the loss functions for the CGAN
    loss_GAN = nn.BCEWithLogitsLoss()
    loss_MSE = nn.MSELoss()
    loss_L1 = nn.L1Loss()

    #Initialize trackers for the loss and image correlation
    Dlossarr =[]
    Glossarr =[]
    MSEarr =[]
    ICarr = []

    scaler = GradScaler(device= "cuda",enabled=torch.cuda.is_available())

    # Training loop
    for epoch in trange(epochs + 1):
        # max_batch_size = train_dataloader.batch_size # This is the configured batch_size from DataLoader
        # Used for initial label tensor sizing.
        # batch_size hyperparameter should be defined globally for this.

        # Define base labels for label smoothing (sized for the maximum possible batch size)
        # Assuming 'batch_size' hyperparameter (e.g., 16) is defined globally and used by DataLoader
        real_label_val = 0.9  # Target for real images
        fake_label_val = 0.1  # Target for fake images
        # These labels are defined once per epoch, then sliced per batch.
        # This is okay as long as the 'batch_size' variable used here is the
        # one configuring the DataLoader (i.e., max_batch_size).
        epoch_valid_labels = torch.full((batch_size, 1), real_label_val, device=device,
                                        dtype=torch.float32)  # Use configured batch_size
        epoch_fake_labels = torch.full((batch_size, 1), fake_label_val, device=device,
                                       dtype=torch.float32)  # Use configured batch_size

        for i, (real_imgs, cond_input) in enumerate(train_dataloader):  # Removed extra parens around train_dataloader

            current_batch_size = real_imgs.size(0)  # Get actual batch size for this iteration
            real_imgs = real_imgs.to(device)
            caps = cond_input.to(device)

            # Using epoch_level labels and slicing them for the current batch
            current_valid_labels_for_batch = epoch_valid_labels[:current_batch_size]
            current_fake_labels_for_batch = epoch_fake_labels[:current_batch_size]


            caps_reshaped = F.interpolate(caps, size=(image_size, image_size), mode='bilinear', align_corners=False)


            # --- Generator step ---
            for _ in range(2):  # Generator trained more frequently
                optimizer_G.zero_grad()

                with autocast(device_type="cuda",dtype=torch.float16,enabled=torch.cuda.is_available()):
                    fake_imgs = G(caps_reshaped)

                    # Adversarial Loss for Generator
                    # G wants D to classify fake_imgs (given condition) as real (target real_label_val)
                    D_output_on_fake = D(caps_reshaped, fake_imgs)
                    adversarial_g_loss = loss_GAN(D_output_on_fake, current_valid_labels_for_batch)

                    # Reconstruction Loss (L1)
                    # Compare generated fake_imgs with the ground truth real_imgs
                    lambda_L1 = Lambda_L1_value  # Hyperparameter: Weight for the L1 loss. Tune this (e.g., 10, 50, 100).
                    l1_reconstruction_loss = loss_L1(fake_imgs, real_imgs)

                    # Total Generator Loss
                    g_loss = adversarial_g_loss + lambda_L1 * l1_reconstruction_loss

                scaler.scale(g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()

                #g_loss.backward()
                #optimizer_G.step()

            # --- Discriminator step ---
            optimizer_D.zero_grad()

            with autocast(device_type="cuda",enabled=torch.cuda.is_available()):
                # Real loss: D evaluates real_imgs (given condition) against real_label_val
                real_loss = loss_GAN(D(caps_reshaped, real_imgs), current_valid_labels_for_batch)

                # Fake loss: D evaluates fake_imgs (given condition) against fake_label_val
                fake_loss = loss_GAN(D(caps_reshaped, fake_imgs.detach()), current_fake_labels_for_batch)

                d_loss = (real_loss + fake_loss) / 2
            #d_loss.backward()
            #torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            #optimizer_D.step()

            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)  # Clip gradients
            scaler.step(optimizer_D)
            scaler.update()

            # Mass loss calculations (metric)
            # Ensure MassCalculator_circular can handle batch tensors if fake_imgs/real_imgs are batches
            # Or loop through them if it expects single images.
            # Assuming it returns a single float (sum over batch or first item) for now.
            # If it returns per-image mass, you might want to average mloss.
            fakeimgmass_batch = MassCalculator_circular(fake_imgs)  # Potential batch sum
            realimgmass_batch = MassCalculator_circular(real_imgs)  # Potential batch sum
            if float(realimgmass_batch) == 0:  # Avoid division by zero
                mloss = float('inf') if float(fakeimgmass_batch) != 0 else 0.0
            else:
                mloss = (abs(float(fakeimgmass_batch) - float(realimgmass_batch)) / float(realimgmass_batch)) * 100

            # IC calculation (metric for the first image in batch)
            with torch.no_grad():
                ic = imagecorrelation_circular(
                    cutoff(real_imgs[0].cpu().numpy(), 0.5, tensorreturn=False),
                    cutoff(fake_imgs[0].cpu().numpy(), 0.5, tensorreturn=False)
                )

            # adjust_learning_rate calls are commented out, StepLR is used instead.
            # adjust_learning_rate(optimizer_G, target_loss, g_loss_total.item()) # Pass scalar total G loss
            # adjust_learning_rate(optimizer_D, target_loss, d_loss.item())     # Pass scalar D loss

        # Save the loss and image correlation values for plotting every epoch
        if epoch % 4 == 0:
            Dlossarr.append(d_loss.item())
            Glossarr.append(g_loss.item())  # Track total generator loss
            # You might also want to track adversarial_g_loss.item() and l1_reconstruction_loss.item() separately
            # G_adv_loss_arr.append(adversarial_g_loss.item())
            # G_L1_loss_arr.append(l1_reconstruction_loss.item())
            MSEarr.append(mloss)  # This is your 'Mass Error'
            with torch.no_grad():
                # IC for the first image of the *last batch* of the epoch
                ic_last_batch_first_img = imagecorrelation_circular(
                    real_imgs[0].cpu().numpy(),
                    # No cutoff here, for consistency with how fake_imgs are often evaluated
                    fake_imgs[0].cpu().numpy()  # No cutoff here, raw output comparison
                )
                ICarr.append(ic_last_batch_first_img)

        if epoch % 10 == 0:
            print(
                f"[{epoch}] D_loss: {d_loss.item():.4f}, G_loss (Total): {g_loss.item():.4f}, "
                # f"G_adv: {adversarial_g_loss.item():.4f}, G_L1: {l1_reconstruction_loss.item():.4f}, " # Optional detailed G loss
                f"Mass Error: {mloss:.2f}%, IC% (last batch, no cutoff): {ic_last_batch_first_img:.2f}%"
            )

        # Visualization
        if epoch % 100 == 0:
            with torch.no_grad():
                # Use g_loss_total for printing

                fig, axs = plt.subplots(3, 8, figsize=(15, 12))
                # Ensure loop is within batch size, or handle if batch size < 8
                num_samples_to_show = min(real_imgs.size(0), 8)  # Use actual batch size, max 8
                for idx in range(num_samples_to_show):  # Changed i to idx to avoid conflict with outer loop i
                    axs[0, idx].imshow(cutoff(real_imgs[idx, 0].cpu(), 0.5), cmap="viridis")
                    axs[0, idx].set_title("Real (Cutoff)")
                    axs[0, idx].axis("off")

                    # Display raw fake image (output of Tanh, in [-1,1])
                    # Or apply cutoff like real for direct comparison: cutoff(fake_imgs[idx,0].cpu(), 0.0, tensorreturn=False)
                    # Or normalize to [0,1] then cutoff: cutoff((fake_imgs[idx,0].cpu()+1)/2, 0.5, tensorreturn=False)
                    axs[1, idx].imshow(fake_imgs[idx, 0].cpu().numpy(), cmap="viridis", vmin=-1,
                                       vmax=1)  # Display raw Tanh output
                    axs[1, idx].set_title("Fake (Raw)")
                    axs[1, idx].axis("off")

                    # Per-image IC and ME for titles (using variables from the last batch of the epoch)
                    # For titles, it's better to recalculate based on the specific 'idx' if possible,
                    # or acknowledge these are for the first item of the last batch if using 'ic' and 'mloss' directly.
                    # The following recalculates for each displayed image:
                    current_ic_value = imagecorrelation_circular(
                        cutoff(real_imgs[idx, 0].cpu().numpy(), 0.5, tensorreturn=False),
                        cutoff(fake_imgs[idx, 0].cpu().numpy(), 0.5, tensorreturn=False)  # Cutoff fake for this IC
                    )
                    # Ensure MassCalculator_circular expects single image tensors here
                    current_mlossgraph = (abs(float(
                        MassCalculator_circular(fake_imgs[idx, 0].unsqueeze(0))) -  # Add batch dim if needed
                                              float(MassCalculator_circular(
                                                  real_imgs[idx, 0].unsqueeze(0)))) /  # Add batch dim if needed
                                          (float(MassCalculator_circular(
                                              real_imgs[idx, 0].unsqueeze(0))) + 1e-9)) * 100  # Avoid div by zero

                    axs[1, idx].set_title(
                        f"Fake (Raw)\nIC(cut): {current_ic_value:.2f}%, ME: {current_mlossgraph:.2f}%", fontsize=6)

                    show_capacitance_image(axs[2, idx], cond_input[idx, 0].cpu().numpy(),
                                           title="Condition")  # Use current cond_input
                plt.tight_layout()
                plt.savefig(f"Outputs/Oval/OvalExamples/epoch_{epoch}.png",dpi=300)
                plt.close(fig)

        # Scheduler step (ensure schedulers are defined if these are uncommented)
        scheduler_G.step()
        scheduler_D.step()

    # Save Generator after training
    output_file = (f"Outputs/Oval/{electrode_count}e{emitting_count}_ECHO_FAN_OVAL_model.pth")
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"File not found: {datafile}")

    torch.save(G.state_dict(), output_file)
    print("Generator model saved.")

    plt.figure(figsize=(15, 5))

    # Plot Glossarr and Dlossarr on the first subplot
    plt.subplot(1, 3, 1)
    plt.plot(Dlossarr, label='Discriminator Loss')
    plt.plot(Glossarr, label='Generator Loss')
    plt.xlabel('Epochs (every 10)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss')

    # Plot MSEarr on the second subplot
    plt.subplot(1, 3, 2)
    plt.plot(MSEarr, label='Mass Error', color='orange')
    plt.xlabel('Epochs (every 10)')
    plt.ylabel('% Error')
    plt.legend()
    plt.title('Mass Error')

    # Plot MSEarr on the second subplot
    plt.subplot(1, 3, 3)
    plt.plot(ICarr, label='Image Correlation', color='blue')
    plt.xlabel('Epochs (every 10)')
    plt.ylabel('% Correlation')
    plt.legend()
    plt.title('Image Correlation')

    plt.tight_layout()
    plt.savefig(overviewsavelocation)




