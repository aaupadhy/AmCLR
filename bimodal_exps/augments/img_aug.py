import torch
import torch.nn.functional as F

def soft_image_blur(image, kernel_size=3, sigma=0.5):
    """
    Apply a soft Gaussian blur to a given image using PyTorch.
    
    Args:
        image (Tensor): Input image tensor with shape [C, H, W].
        kernel_size (int): Size of the Gaussian kernel (must be odd, typically 3 or 5 for soft blur).
        sigma (float): Standard deviation of the Gaussian kernel (lower values = softer blur).
    
    Returns:
        Tensor: Blurred image tensor with shape [C, H, W].
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    # Create Gaussian kernel
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()  # Normalize

    # Create 2D Gaussian kernel
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, kernel_size, kernel_size]
    
    # Convolve each channel of the image
    padding = kernel_size // 2
    blurred_image = F.conv2d(
        image.unsqueeze(0),  # Add batch dimension
        kernel_2d.expand(image.shape[0], 1, kernel_size, kernel_size),  # Apply kernel to each channel
        padding=padding,
        groups=image.shape[0]  # Separate kernels for each channel
    )
    return blurred_image.squeeze(0)
