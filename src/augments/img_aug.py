import torch
import torch.nn.functional as F
import random

def soft_image_blur(images, kernel_size=3, sigma=0.5):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    x = torch.arange(kernel_size, dtype=images.dtype, device=images.device) - (kernel_size - 1) / 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    C = images.shape[1]
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size).to(images.device)
    padding = kernel_size // 2
    return F.conv2d(images, kernel_2d, padding=padding, groups=C)

def random_flip(images):
    if random.random() > 0.5:
        images = torch.flip(images, dims=[3])
    if random.random() > 0.5:
        images = torch.flip(images, dims=[2])
    return images

def add_random_noise(images, noise_level=0.05):
    noise = torch.randn_like(images) * noise_level
    return torch.clamp(images + noise, 0, 1)

def random_brightness(images, brightness_range=(0.8, 1.2)):
    factor = torch.tensor(random.uniform(*brightness_range), device=images.device)
    return torch.clamp(images * factor, 0, 1)

def random_contrast(images, contrast_range=(0.8, 1.2)):
    mean = torch.mean(images, dim=(2, 3), keepdim=True)
    factor = torch.tensor(random.uniform(*contrast_range), device=images.device)
    return torch.clamp((images - mean) * factor + mean, 0, 1)

def random_sharpen(images, alpha=0.5):
    alpha = torch.tensor(alpha, device=images.device)
    blurred = soft_image_blur(images)
    return torch.clamp(images + alpha * (images - blurred), 0, 1)

# def random_rotate(images, degrees=5):
#     angle = random.uniform(-degrees, degrees) * torch.pi / 180
#     angle_tensor = torch.tensor(angle, device=images.device)
#     cos_theta = torch.cos(angle_tensor)
#     sin_theta = torch.sin(angle_tensor)
#     theta = torch.tensor([[cos_theta, -sin_theta, 0],
#                           [sin_theta, cos_theta, 0]], device=images.device).unsqueeze(0)
#     grid = F.affine_grid(theta, images.size(), align_corners=False)
#     return F.grid_sample(images, grid, align_corners=False)

def center_crop(images, crop_fraction=0.9):
    _, _, H, W = images.shape
    new_H, new_W = int(H * crop_fraction), int(W * crop_fraction)
    start_H, start_W = (H - new_H) // 2, (W - new_W) // 2
    return images[:, :, start_H:start_H + new_H, start_W:start_W + new_W]

def pad_to_original(images, crop_fraction=0.9):
    cropped = center_crop(images, crop_fraction)
    _, _, H, W = images.shape
    padded = F.pad(cropped, [0, W - cropped.shape[3], 0, H - cropped.shape[2]], mode='constant', value=0)
    return padded

# def random_flip_rotation(images):
#     rotated = random_rotate(images, degrees=5)
#     flipped = random_flip(rotated)
#     return flipped

AUGMENTATIONS = [
    soft_image_blur,
    random_flip,
    add_random_noise,
    random_brightness,
    random_contrast,
    random_sharpen,
    center_crop,
    pad_to_original,
    # random_flip_rotation
]

def augmenter(images):
    aug_func = random.choice(AUGMENTATIONS)
    return aug_func(images)
