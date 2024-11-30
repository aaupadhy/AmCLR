import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
from img_aug import augmenter  

output_dir = "plots/"
os.makedirs(output_dir, exist_ok=True)

# Load CIFAR-10 dataset
transform = transforms.ToTensor()
cifar10 = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Select a few images for testing
num_images = 5
test_images = [cifar10[i][0].unsqueeze(0) for i in range(num_images)]  # Only the image tensors

# Apply augmentations and save results
for idx, image_tensor in enumerate(test_images):
    # Save original image
    original_numpy = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    plt.figure()
    plt.imshow(original_numpy)
    plt.axis('off')
    plt.title(f"Original Image {idx + 1}")
    plt.savefig(os.path.join(output_dir, f"original_image_{idx + 1}.png"))
    plt.close()

    # Apply augmentations and save results
    for aug_idx in range(5):  # Number of augmentations to apply
        augmented_image = augmenter(image_tensor)
        augmented_numpy = augmented_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        plt.figure()
        plt.imshow(augmented_numpy)
        plt.axis('off')
        plt.title(f"Image {idx + 1}, Augmentation {aug_idx + 1}")
        plt.savefig(os.path.join(output_dir, f"image_{idx + 1}_aug_{aug_idx + 1}.png"))
        plt.close()

print(f"Original and augmented images saved in {output_dir}")
