import torch
import torch.nn.functional as F

def soft_image_blur(image, kernel_size=3, sigma=0.5):
    """
    Apply a soft Gaussian blur to a given image tensor using PyTorch.
    
    Args:
        image (Tensor): Input image tensor with shape [B, C, H, W].
        kernel_size (int): Size of the Gaussian kernel (must be odd, typically 3 or 5 for soft blur).
        sigma (float): Standard deviation of the Gaussian kernel (lower values = softer blur).
    
    Returns:
        Tensor: Blurred image tensor with the same shape as the input.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    x = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - (kernel_size - 1) / 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()  # Normalize

    # Create 2D Gaussian kernel
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, kernel_size, kernel_size]

    # Expand kernel for each channel
    C = image.shape[1]
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)  # Shape [C, 1, kernel_size, kernel_size]

    # Ensure kernel is on the same device as the image
    kernel_2d = kernel_2d.to(image.device)

    # Convolve each channel of the image
    padding = kernel_size // 2
    blurred_image = F.conv2d(
        image,  # Input image tensor with shape [B, C, H, W]
        kernel_2d,  # Gaussian kernel
        padding=padding,
        groups=C  # Separate kernel per channel
    )
    return blurred_image


