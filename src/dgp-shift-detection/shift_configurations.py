import numpy as np


registered_configurations = {}


def register_configuration(configuration_name: str):
    def decorator(func):
        registered_configurations[configuration_name] = func
        return func

    return decorator


@register_configuration("gaussian_noise")
def get_gaussian_noise_configurations(dataset_name: str):
    if dataset_name == "mnist":
        sigmas = np.array([25.0, 40.0, 55.0, 70.0, 85.0, 100.0]) / 255.0
    elif dataset_name == "fashion-mnist":
        sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]) / 255.0
    elif dataset_name == "cifar10":
        sigmas = np.array([30.0, 60.0, 85.0, 100.0, 120.0, 140.0]) / 255.0
    else:
        raise ValueError(
            f"Gaussian Noise: No configuration defined for dataset: {dataset_name}"
        )

    return [{"sigma": sigma} for sigma in sigmas.tolist()]


@register_configuration("salt_pepper_noise")
def get_salt_pepper_noise_configurations(dataset_name: str):
    if dataset_name in ["mnist", "fashion-mnist", "cifar10"]:
        noise_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        raise ValueError(
            f"Salt Pepper Noise: No configuration defined for dataset: {dataset_name}"
        )

    return [{"noise_ratio": noise_ratio} for noise_ratio in noise_ratios]


@register_configuration("gaussian_blur")
def get_gaussian_blur_configurations(dataset_name: str):
    if dataset_name == "mnist" or dataset_name == "fashion-mnist":
        sigmas = [0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
        kernel_sizes = [(3, 3)] * len(sigmas)
    elif dataset_name == "cifar10":
        sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        kernel_sizes = [(3, 3), (3, 5), (5, 5), (5, 7), (7, 7), (9, 9)]
    else:
        raise ValueError(
            f"Gaussian Blur: No configuration defined for dataset: {dataset_name}"
        )

    configurations = [
        {"sigma": sigma, "kernel_size": kernel_size}
        for sigma, kernel_size in zip(sigmas, kernel_sizes)
    ]

    return configurations


@register_configuration("image_transform")
def get_image_transform_configurations(dataset_name: str):
    if dataset_name == "mnist":
        rotation_range = [1, 5, 7, 10, 12.5, 25]
        width_shift_range = [0.01, 0.05, 0.075, 0.1, 0.12, 0.15]
        height_shift_range = [0.01, 0.05, 0.075, 0.1, 0.12, 0.15]
        shear_range = [0, 0.01, 0.02, 0.04, 0.06, 0.12]
        zoom_range = [0, 0.01, 0.02, 0.04, 0.06, 0.12]
        horizontal_flip = [False, False, False, False, False, False]
        vertical_flip = [False, False, False, False, False, False]

    elif dataset_name == "fashion-mnist":
        rotation_range = [1, 5, 7, 10, 12.5, 15]
        width_shift_range = [0.01, 0.05, 0.075, 0.1, 0.12, 0.15]
        height_shift_range = [0.01, 0.05, 0.075, 0.1, 0.12, 0.15]
        shear_range = [0, 0.01, 0.02, 0.04, 0.06, 0.08]
        zoom_range = [0, 0.01, 0.02, 0.04, 0.06, 0.08]
        horizontal_flip = [False, False, False, False, False, False]
        vertical_flip = [False, False, False, False, False, False]

    elif dataset_name == "cifar10":
        rotation_range = [20, 30, 35, 40, 45, 50]
        width_shift_range = [0.175, 0.2, 0.225, 0.25, 0.31, 0.375]
        height_shift_range = [0.175, 0.2, 0.225, 0.25, 0.31, 0.375]
        shear_range = [0.1, 0.12, 0.15, 0.18, 0.23, 0.27]
        zoom_range = [0.1, 0.12, 0.15, 0.18, 0.23, 0.27]
        horizontal_flip = [False, False, False, False, False, False]
        vertical_flip = [False, False, False, False, False, False]

    else:
        raise ValueError(
            f"Image Transform: No configuration defined for dataset: {dataset_name}"
        )

    configurations = [
        {
            "rotation_range": rotation_range[i],
            "width_shift_range": width_shift_range[i],
            "height_shift_range": height_shift_range[i],
            "shear_range": shear_range[i],
            "zoom_range": zoom_range[i],
            "horizontal_flip": horizontal_flip[i],
            "vertical_flip": vertical_flip[i],
        }
        for i in range(len(rotation_range))
    ]

    return configurations


@register_configuration("uniform_noise")
def get_uniform_noise_configurations(dataset_name: str):
    return get_gaussian_noise_configurations(dataset_name)


@register_configuration("pixel_shuffle")
def get_pixel_shuffle_configurations(dataset_name: str):
    if dataset_name in ["mnist", "fashion-mnist", "cifar10"]:
        kernel_sizes = [3, 5, 7, 11, 13, 17]
    else:
        raise ValueError("No configuration defined for dataset: {dataset_name}")
    return [{"kernel_size": kernel_size} for kernel_size in kernel_sizes]


@register_configuration("pixel_dropout")
def get_pixel_dropout_configurations(dataset_name: str):
    if dataset_name in ["mnist", "fashion-mnist", "cifar10"]:
        noise_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        raise ValueError(
            f"Pixel Dropout: No configuration defined for dataset: {dataset_name}"
        )

    return [{"p": noise_ratio} for noise_ratio in noise_ratios]


def get_shift_configurations(dataset_name: str, shift_name: str):
    if shift_name in registered_configurations:
        return registered_configurations[shift_name](dataset_name)
    else:
        raise ValueError(f"No configuration defined for shift: {shift_name}")
