from PIL import Image, ImageOps, ImageFilter, ImageDraw
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt


class TableExtractor:
    def __init__(self, image_path):
        self.image_path = image_path

    def read_image(self):
        self.image = Image.open(self.image_path)

    def store_process_image(self, output_path, image):
        image.save(output_path)

    def convert_image_to_grayscale(self):
        self.grayscale_image = ImageOps.grayscale(self.image)

    def threshold_image(self):
        threshold_value = 100  # Adjust the threshold value as needed
        self.thresholded_image = self.grayscale_image.point(
            lambda p: 255 if p > threshold_value else 0)

    def invert_image(self):
        self.inverted_image = ImageOps.invert(self.thresholded_image)

    def dilate_image(self):
        kernel = ImageFilter.Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.dilated_image = self.inverted_image.filter(kernel)

    def find_contours(self):
        labeled_image = measure.label(np.array(self.dilated_image))
        regions = measure.regionprops(labeled_image)
        self.contours = [region.bbox for region in regions]
        self.image_with_all_contours = self.image.copy()
        self.draw_contours()

    def draw_contours(self):
        draw = ImageDraw.Draw(self.image_with_all_contours)
        for contour in self.contours:
            min_row, min_col, max_row, max_col = contour
            draw.rectangle([min_col, min_row, max_col, max_row],
                           outline=(0, 255, 0), width=3)

    def execute(self):
        self.read_image()
        self.store_process_image(
            "./uploads/0_original.jpg", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image(
            "./uploads/1_grayscaled.jpg", self.grayscale_image)
        self.threshold_image()
        self.store_process_image(
            "./uploads/2_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image(
            "./uploads/3_inverteded.jpg", self.inverted_image)
        self.dilate_image()
        self.store_process_image(
            "./uploads/4_dialateded.jpg", self.dilated_image)
        self.find_contours()
        self.store_process_image(
            "./uploads/5_all_contours.jpg", self.image_with_all_contours)
