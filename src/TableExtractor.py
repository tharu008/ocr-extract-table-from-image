from PIL import Image, ImageOps, ImageFilter, ImageDraw
from skimage import measure
from skimage.morphology import rectangle, binary_erosion, binary_dilation
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
from shapely.geometry import Polygon
from shapely.ops import unary_union


# def perimeter(polygon):
#    return sum(polygon.length for polygon in polygon.geoms)


class TableExtractor:

    def __init__(self, image_path):
        self.image_path = image_path
        self.rectangular_contours = []

    # Read image
    def read_image(self):
        self.image = Image.open(self.image_path)

    # Store processed image
    def store_process_image(self, output_path, image):
        image.save(output_path)

    # Convert image to grayscale
    def convert_image_to_grayscale(self):
        self.grayscale_image = ImageOps.grayscale(self.image)

    # Threshold image
    def threshold_image(self):
        threshold_value = 150  # Adjust the threshold value as needed
        self.thresholded_image = self.grayscale_image.point(
            lambda p: 255 if p > threshold_value else 0)

    # Invert image
    def invert_image(self):
        self.inverted_image = ImageOps.invert(self.thresholded_image)

    # Extracting vertical lines
    # erosion vertical lines
    def v_erosion_image(self, iterations=1):
        # Convert PIL Image to NumPy array
        image_array = np.array(self.inverted_image)
        # Create a vertical erosion kernel
        vertical_kernel = rectangle(5, 1)
        # Perform vertical erosion iteratively on the image array
        eroded_image_array = image_array.copy()  # Make a copy to preserve original
        for _ in range(iterations):
            eroded_image_array = binary_erosion(
                eroded_image_array, vertical_kernel)
        # Convert the eroded image array back to PIL Image
        self.v_eroded_image = Image.fromarray(eroded_image_array)

    # dilation vertical lines
    def v_dilation_image(self, iterations=5):
        image_array = np.array(self.v_eroded_image)
        vertical_kernel = rectangle(5, 1)
        dilated_image_array = image_array.copy()  # Make a copy to preserve original
        for _ in range(iterations):
            dilated_image_array = binary_dilation(
                dilated_image_array, vertical_kernel)
        self.v_dilated_image = Image.fromarray(dilated_image_array)

    # extracting horizontal lines
    # erosion horizontal lines
    def h_erosion_image(self, iterations=5):
        image_array = np.array(self.inverted_image)
        horizontal_kernel = rectangle(1, 5)
        eroded_image_array = image_array.copy()  # Make a copy to preserve original
        for _ in range(iterations):
            eroded_image_array = binary_erosion(
                eroded_image_array, horizontal_kernel)
        self.h_eroded_image = Image.fromarray(eroded_image_array)

    # dilation horizontal lines
    def h_dilation_image(self, iterations=5):
        image_array = np.array(self.h_eroded_image)
        horizontal_kernel = rectangle(1, 5)
        dilated_image_array = image_array.copy()  # Make a copy to preserve original
        for _ in range(iterations):
            dilated_image_array = binary_dilation(
                dilated_image_array, horizontal_kernel)
        self.h_dilated_image = Image.fromarray(dilated_image_array)

    # Blending vertical and horizontal lines
    def blend_images(self, weight1, weight2, gamma=0.0):
        v_dilated_image_array = np.array(self.v_dilated_image)
        h_dilated_image_array = np.array(self.h_dilated_image)

        blended_array = (weight1 * v_dilated_image_array +
                         weight2 * h_dilated_image_array + gamma).astype(np.uint8)

        # Normalize the blended array to [0, 255]
        blended_array = ((blended_array - blended_array.min()) /
                         (blended_array.max() - blended_array.min()) * 255).astype(np.uint8)

        self.blended_image = Image.fromarray(blended_array)

    '''
    def erosion_blended_image(self, iterations=1):
        image_array = np.array(self.blended_image)
        kernel = rectangle(2, 2)
        eroded_image_array = image_array.copy()  # Make a copy to preserve original
        for _ in range(iterations):
            eroded_image_array = binary_erosion(
                eroded_image_array, kernel)
        self.eroded_blended_image = Image.fromarray(eroded_image_array)

    
    def dilate_image(self):
        kernel = ImageFilter.Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.dilated_image = self.inverted_image.filter(kernel)
    '''

    # Threshold blended image
    def threshold_blended_image(self):
        threshold_value = 120  # Adjust the threshold value as needed
        self.thresh_blended_image = self.blended_image.point(
            lambda p: 255 if p > threshold_value else 0)

    # Find contours
    def find_contours(self, threshold_value=128):
        image_array = np.array(self.thresh_blended_image)
        binary_image = image_array > threshold_value
        contours = measure.find_contours(binary_image, 0.5)

        # Create an empty PIL image for visualization
        contour_image = self.image.copy()
        draw = ImageDraw.Draw(contour_image)

        # Draw the contours on the image
        for contour in contours:
            contour = np.round(contour).astype(int)
            draw.line(
                list(zip(contour[:, 1], contour[:, 0])), fill='green', width=2)

        self.contour_img = contour_image

    '''
    def find_contours(self):
        labeled_image = measure.label(np.array(self.thresh_blended_image))
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

    '''

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
        self.v_erosion_image(iterations=5)
        self.store_process_image(
            "./uploads/4_vertical_eroded.jpg", self.v_eroded_image)
        self.v_dilation_image(iterations=5)
        self.store_process_image(
            "./uploads/5_vertical_dilated.jpg", self.v_dilated_image)
        self.h_erosion_image(iterations=5)
        self.store_process_image(
            "./uploads/6_horizontal_eroded.jpg", self.h_eroded_image)
        self.h_dilation_image(iterations=5)
        self.store_process_image(
            "./uploads/7_horizontal_dilated.jpg", self.h_dilated_image)
        self.blend_images(1, 1)
        self.store_process_image(
            "./uploads/8_blended.jpg", self.blended_image)
        # self.erosion_blended_image(iterations=1)
        # self.store_process_image(
        #    "./uploads/9_eroded_blended.jpg", self.eroded_blended_image)
        self.threshold_blended_image()
        self.store_process_image(
            "./uploads/10_thresholded_blended.jpg", self.thresh_blended_image)

        # self.store_process_image(
    #     "./uploads/_dialateded.jpg", self.dilated_image)
        self.find_contours(threshold_value=128)
        self.store_process_image(
            "./uploads/11_all_contours.jpg", self.contour_img)
