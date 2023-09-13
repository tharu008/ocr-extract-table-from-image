from PIL import Image, ImageOps
from skimage.morphology import rectangle, binary_erosion, binary_dilation
import numpy as np
import cv2


class TableExtractor:

    def __init__(self, image_path):
        self.image_path = image_path
        self.rectangular_contours = []

    # Read image - PIL Image
    def read_image(self):
        self.image = Image.open(self.image_path)

    # Store processed image - PIL Image
    def store_process_image(self, output_path, image):
        image.save(output_path)

    # Convert image to grayscale - PIL Image
    def convert_image_to_grayscale(self):
        self.grayscale_image = ImageOps.grayscale(self.image)

    # Threshold image - PIL Image
    def threshold_image(self):
        threshold_value = 150  # Adjust the threshold value as needed
        self.thresholded_image = self.grayscale_image.point(
            lambda p: 255 if p > threshold_value else 0)

    # Invert image - PIL Image
    def invert_image(self):
        self.inverted_image = ImageOps.invert(self.thresholded_image)

    # Extracting vertical lines - skimage
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

    # dilation vertical lines - skimage
    def v_dilation_image(self, iterations=5):
        image_array = np.array(self.v_eroded_image)
        vertical_kernel = rectangle(5, 1)
        dilated_image_array = image_array.copy()  # Make a copy to preserve original
        for _ in range(iterations):
            dilated_image_array = binary_dilation(
                dilated_image_array, vertical_kernel)
        self.v_dilated_image = Image.fromarray(dilated_image_array)

    # extracting horizontal lines
    # erosion horizontal lines - skimage
    def h_erosion_image(self, iterations=5):
        image_array = np.array(self.inverted_image)
        horizontal_kernel = rectangle(1, 5)
        eroded_image_array = image_array.copy()  # Make a copy to preserve original
        for _ in range(iterations):
            eroded_image_array = binary_erosion(
                eroded_image_array, horizontal_kernel)
        self.h_eroded_image = Image.fromarray(eroded_image_array)

    # dilation horizontal lines - skimage
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
        # numpy array used for blending (calculate on image data)
        blended_array = (weight1 * v_dilated_image_array +
                         weight2 * h_dilated_image_array + gamma).astype(np.uint8)

        # Normalize the blended array to [0, 255]
        blended_array = ((blended_array - blended_array.min()) /
                         (blended_array.max() - blended_array.min()) * 255).astype(np.uint8)
        # PIL image used for visualization(convert np to PIL)
        self.blended_image = Image.fromarray(blended_array)

    # Threshold blended image - PIL Image
    def threshold_blended_image(self):
        threshold_value = 120  # Adjust the threshold value as needed
        self.thresh_blended_image = self.blended_image.point(
            lambda p: 255 if p > threshold_value else 0)

    # Find contours - cv2
    def find_contours(self):
        img = np.array(self.thresh_blended_image)
        self.contours, self.hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.image_with_all_contours = self.image.copy()
        image_array = np.array(self.image_with_all_contours)

        cv2.drawContours(image_array,
                         self.contours, -1, (0, 255, 0), 3)
        self.image_with_all_contours = Image.fromarray(image_array)

    def filter_contours_and_leave_only_rectangles(self):
        self.rectangular_contours = []
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)
        self.image_with_only_rectangular_contours = self.image.copy()
        image_array = np.array(self.image_with_only_rectangular_contours)

        cv2.drawContours(image_array,
                         self.rectangular_contours, -1, (0, 255, 0), 3)
        self.image_with_only_rectangular_contours = Image.fromarray(
            image_array)

    def find_largest_contour_by_area(self):
        max_area = 0
        self.contour_with_max_area = None
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour
        self.image_with_contour_with_max_area = self.image.copy()
        image_array = np.array(self.image_with_only_rectangular_contours)

        cv2.drawContours(image_array, [
                         self.contour_with_max_area], -1, (0, 255, 0), 3)
        self.image_with_contour_with_max_area = Image.fromarray(image_array)

    def order_points_in_the_contour_with_max_area(self):
        self.contour_with_max_area_ordered = self.order_points(
            self.contour_with_max_area)
        self.image_with_points_plotted = self.image.copy()
        for point in self.contour_with_max_area_ordered:
            point_coordinates = (int(point[0]), int(point[1]))
            image_array = np.array(self.image_with_points_plotted)

            self.image_with_points_plotted = cv2.circle(
                image_array, point_coordinates, 10, (0, 0, 255), -1)
            self.image_with_points_plotted = Image.fromarray(image_array)

    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def calculate_new_width_and_height_of_image(self):
        existing_image_width, existing_image_height = self.image.size
        existing_image_width_reduced_by_10_percent = int(
            existing_image_width * 0.9)

        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(
            self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(
            self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = distance_between_top_left_and_bottom_left / \
            distance_between_top_left_and_top_right

        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)

    def calculateDistanceBetween2Points(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def apply_perspective_transform(self):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [
                          self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        # Convert the PIL image to a NumPy array
        src_image = np.array(self.image)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_array = cv2.warpPerspective(
            src_image, matrix, (self.new_image_width, self.new_image_height))
        self.perspective_corrected_image = Image.fromarray(img_array)

    def add_10_percent_padding(self):
        image_width, image_height = self.perspective_corrected_image.size
        padding = int(image_height * 0.1)
        # Create a white background image with the desired dimensions
        padded_image = Image.new(
            'RGB', (image_width + 2 * padding, image_height + 2 * padding), (255, 255, 255))
        # Paste the perspective corrected image onto the white background with padding
        padded_image.paste(self.perspective_corrected_image,
                           (padding, padding))
        self.perspective_corrected_image_with_padding = padded_image

    def execute(self):
        self.read_image()
        self.store_process_image(
            "./uploads/TableExtractor/0_original.jpg", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image(
            "./uploads/TableExtractor/1_grayscaled.jpg", self.grayscale_image)
        self.threshold_image()
        self.store_process_image(
            "./uploads/TableExtractor/2_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image(
            "./uploads/TableExtractor/3_inverteded.jpg", self.inverted_image)
        self.v_erosion_image(iterations=5)
        self.store_process_image(
            "./uploads/TableExtractor/4_vertical_eroded.jpg", self.v_eroded_image)
        self.v_dilation_image(iterations=5)
        self.store_process_image(
            "./uploads/TableExtractor/5_vertical_dilated.jpg", self.v_dilated_image)
        self.h_erosion_image(iterations=5)
        self.store_process_image(
            "./uploads/TableExtractor/6_horizontal_eroded.jpg", self.h_eroded_image)
        self.h_dilation_image(iterations=5)
        self.store_process_image(
            "./uploads/TableExtractor/7_horizontal_dilated.jpg", self.h_dilated_image)
        self.blend_images(1, 1)
        self.store_process_image(
            "./uploads/TableExtractor/8_blended.jpg", self.blended_image)
        self.threshold_blended_image()
        self.store_process_image(
            "./uploads/TableExtractor/9_thresholded_blended.jpg", self.thresh_blended_image)
        self.find_contours()
        self.store_process_image(
            "./uploads/TableExtractor/10_all_contours.jpg", self.image_with_all_contours)
        self.filter_contours_and_leave_only_rectangles()
        self.store_process_image(
            "./uploads/TableExtractor/11_only_rectangular_contours.jpg", self.image_with_only_rectangular_contours)
        self.find_largest_contour_by_area()
        self.store_process_image(
            "./uploads/TableExtractor/12_contour_with_max_area.jpg", self.image_with_contour_with_max_area)
        self.order_points_in_the_contour_with_max_area()
        self.store_process_image(
            "./uploads/TableExtractor/13_with_4_corner_points_plotted.jpg", self.image_with_points_plotted)
        self.calculate_new_width_and_height_of_image()
        self.apply_perspective_transform()
        self.store_process_image(
            "./uploads/TableExtractor/14_perspective_corrected.jpg", self.perspective_corrected_image)
        self.add_10_percent_padding()
        self.store_process_image("./uploads/TableExtractor/15_perspective_corrected_with_padding.jpg",
                                 self.perspective_corrected_image_with_padding)
        return self.perspective_corrected_image_with_padding
