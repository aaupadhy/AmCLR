import torch
import torch.nn.functional as F

def soft_image_blur(images, kernel_size=3, sigma=0.5):
    """
    Apply a soft Gaussian blur to a batch of images using PyTorch.

    Args:
        images (Tensor): Batch of input images with shape [B, C, H, W].
        kernel_size (int): Size of the Gaussian kernel (must be odd, typically 3 or 5 for soft blur).
        sigma (float): Standard deviation of the Gaussian kernel (lower values = softer blur).

    Returns:
        Tensor: Blurred batch of images with the same shape as the input.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=images.dtype, device=images.device) - (kernel_size - 1) / 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()  # Normalize

    # Create 2D Gaussian kernel
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, kernel_size, kernel_size]

    # Expand kernel for each channel
    C = images.shape[1]
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)  # Shape [C, 1, kernel_size, kernel_size]

    # Ensure kernel is on the same device as the images
    kernel_2d = kernel_2d.to(images.device)

    # Convolve each channel of the image
    padding = kernel_size // 2
    blurred_images = F.conv2d(
        images,  # Input image tensor with shape [B, C, H, W]
        kernel_2d,  # Gaussian kernel
        padding=padding,
        groups=C  # Separate kernel per channel
    )
    return blurred_images
