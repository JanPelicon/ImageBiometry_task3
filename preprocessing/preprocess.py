import cv2
import numpy as np
from skimage.filters import sobel


class Preprocess:

    @staticmethod
    def rgb_to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def normalization(image):  # GRAY
        min_val = np.min(image)
        max_val = np.max(image)
        normalized = (image - min_val) * (255 / (max_val - min_val))
        return normalized.astype(np.uint8)

    @staticmethod
    def resize(image, size):
        return cv2.resize(image, size)

    def pipeline(self, image, size):
        image = self.rgb_to_gray(image)
        image = self.normalization(image)
        image = self.resize(image, (size, size))
        return image

    @staticmethod
    def sobel(image):
        image = sobel(image)
        return image
