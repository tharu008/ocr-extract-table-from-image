import numpy as np
import gaussian_filter as gf
import cv2


def apply_adaptive_threshold_gaussian(image, block_size, sigma, c):
    output_image = np.zeros_like(image)
    height, width = image.shape[:2]
    size = block_size // 2

    gaussian_filtered = gf.gaussian_filter(image, gf.gaussian_kernel(block_size, sigma))

    local_threshold = gaussian_filtered - c

    # Apply the threshold to the input image
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > local_threshold] = 255

    return binary


image = cv2.imread("0.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = gf.gaussian_filter(gray, gf.gaussian_kernel(3, 1))
thresholded_image = apply_adaptive_threshold_gaussian(filtered, 27, 10, 5)
cv2.imwrite("output.jpg", thresholded_image)
