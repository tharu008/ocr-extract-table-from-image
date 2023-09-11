import cv2
import numpy as np
from PIL import Image, ImageOps


class OcrToTableTool:
    def __init__(self, image):
        self.image = image

    # Read image - PIL Image
    def read_image(self):
        pass

    def execute(self):
        self.read_image()
