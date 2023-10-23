import torch
import numpy as np

from typing import Tuple
from torchvision.transforms import GaussianBlur
from keras.preprocessing.image import ImageDataGenerator


registered_shifts = {}


def register_shift(shift_name: str):
    def decorator(func):
        registered_shifts[shift_name] = func
        return func

    return decorator


@register_shift("gaussian_noise")
def gaussian_noise(x: np.ndarray, y: np.ndarray, sigma: float):
    # Generate noise
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)

    # Apply noise
    x_noise = x + noise
    x_noise = np.clip(x_noise, 0.0, 1.0)

    return x_noise, y


@register_shift("salt_pepper_noise")
def salt_pepper_noise(x: np.ndarray, y: np.ndarray, noise_ratio: float):
    # Generate noise
    noise = np.random.choice(
        [-1, 0, 1], size=x.shape, p=[noise_ratio / 2, 1 - noise_ratio, noise_ratio / 2]
    )

    # Apply noise
    x_noise = x.copy()
    x_noise = np.where(noise == -1, 0.0, x_noise)
    x_noise = np.where(noise == 1, 1.0, x_noise)

    return x_noise, y


@register_shift("gaussian_blur")
def gaussian_blur(
    x: np.ndarray, y: np.ndarray, sigma: float, kernel_size: Tuple[int, int] = (3, 3)
):
    """
    Inpsired by MAGDIFF Code
    https://github.com/hensel-f/MAGDiff_experiments/blob/main/utils/utils_shift_functions.py
    """
    # Convert to torch
    x = torch.from_numpy(x).float()

    # Check dimensions
    if x.ndim == 3:
        x = x.unsqueeze(1)
    assert x.ndim == 4

    # Instantiate Gaussian Blur
    gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    # Apply Gaussian Blur
    x_blur = np.stack([gaussian_blur(img).numpy() for img in x])

    return x_blur, y


@register_shift("image_transform")
def image_transform(x: np.ndarray, y: np.ndarray, **transform_parameters):
    """
    Inpsired by MAGDIFF Code
    https://github.com/hensel-f/MAGDiff_experiments/blob/main/utils/utils_shift_functions.py

    Example for `transform_parameters`:
    transform_parameters={
        "rotation_range": 100 * sigma,
        "zoom_range": sigma,
        "shear_range": sigma,
        "vertical_flip": sigma > 0.5,
        "width_shift_range": sigma / 2.,
        "height_shift_range": sigma / 2.,
    }
    """
    # Convert to torch
    x = torch.from_numpy(x).float()

    # Check dimensions
    if x.ndim == 3:
        x = x.unsqueeze(1)
    assert x.ndim == 4

    # Instantiate ImageDataGenerator
    transform_parameters.update(
        {
            "fill_mode": "nearest",
            "data_format": "channels_first",
        }
    )
    image_data_generator = ImageDataGenerator(**transform_parameters)

    # Apply ImageDataGenerator
    x_transformed = image_data_generator.flow(x, batch_size=len(x), shuffle=False)
    x_transformed = next(x_transformed)

    return x_transformed, y


@register_shift("uniform_noise")
def uniform_noise(x: np.ndarray, y: np.ndarray, sigma: float):
    # Generate noise
    noise = np.random.uniform(low=-sigma, high=sigma, size=x.shape)

    # Apply noise
    x_noise = x + noise
    x_noise = np.clip(x_noise, 0.0, 1.0)

    return x_noise, y


@register_shift("pixel_shuffle")
def pixel_shuffle(x: np.ndarray, y: np.ndarray, kernel_size: int):
    # Check dimensions
    if x.ndim == 3:
        x = x[:, np.newaxis, :, :]
    assert x.ndim == 4

    # Get dimensions
    _, channels, height, width = x.shape

    # Iterate over patches
    x_shuffle = []
    for img in x:
        img_shuffle = img.copy()
        for x_start in range(0, height - kernel_size):
            for y_start in range(0, width - kernel_size):
                # Get end indices
                x_end = x_start + kernel_size
                y_end = y_start + kernel_size

                patch = img[:, x_start:x_end, y_start:y_end]
                patch_height, patch_width = patch.shape[1:]
                num_patch_pixels = patch_height * patch_width
                patch = patch.reshape(channels, -1)
                permutation = np.random.permutation(num_patch_pixels)
                patch = patch[:, permutation]
                patch = patch.reshape(channels, patch_height, patch_width)

                img_shuffle[:, x_start:x_end, y_start:y_end] = patch
        x_shuffle.append(img_shuffle)

    x_shuffle = np.stack(x_shuffle)
    return x_shuffle, y


@register_shift("pixel_dropout")
def pixel_dropout(x: np.ndarray, y: np.ndarray, p: float):
    # Check dimensions
    if x.ndim == 3:
        x = x[:, np.newaxis, :, :]
    assert x.ndim == 4

    # Make dropout mask
    assert 0.0 <= p <= 1.0
    num_images, channels, height, width = x.shape
    dropout_mask = np.random.binomial(n=1, p=p, size=(num_images, height, width))
    dropout_mask = np.repeat(dropout_mask[:, np.newaxis, :, :], channels, axis=1)
    dropout_mask = dropout_mask.astype(bool)

    # Apply dropout mask
    x_dropout = x.copy()
    x_dropout[dropout_mask] = 0.0

    # Return
    return x_dropout, y
