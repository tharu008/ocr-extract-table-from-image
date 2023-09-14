import numpy as np


def gaussian_kernel(size, sigma=1):
    kernel = np.zeros((size, size))
    center = size // 2
    x, y = np.mgrid[-center : center + 1, -center : center + 1]
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return kernel / np.sum(kernel)


def gaussian_filter(image, kernel):
    kernel_size = kernel.shape[0]

    size = kernel_size // 2
    image_pad = np.pad(
        image,
        size,
        mode="constant",
        constant_values=0,
    ).astype(np.float32)
    image_h, image_w = image_pad.shape[:2]

    for i in range(size, image_h - size):
        for j in range(size, image_w - size):
            roi = image_pad[i - size : i + size + 1, j - size : j + size + 1]
            image_pad[i, j] = np.sum(roi * kernel)

    return image_pad[size:-size, size:-size]
