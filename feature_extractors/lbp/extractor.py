import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBP:
	def __init__(self, points=8, radius=1, size=64, window_stride=8, bins=8):
		self.num_points = points * radius
		self.radius = radius
		self.size = size
		self.window_stride = window_stride
		self.bins = bins

	def extract_lbp(self, image):
		lbp = local_binary_pattern(image, self.num_points, self.radius)
		return lbp

	def extract_1d(self, image):
		lbp = self.extract_lbp(image)
		lbp = lbp.ravel()
		return lbp

	def extract_hist(self, image):
		lbp = self.extract_lbp(image)
		main_histogram = np.array([], dtype=int)
		for y in range(0, self.size, self.window_stride):
			for x in range(0, self.size, self.window_stride):
				subimage = lbp[y:y+self.window_stride, x:x+self.window_stride]
				subhistogram, _ = np.histogram(subimage, bins=self.bins)
				main_histogram = np.concatenate((main_histogram, subhistogram))
		main_histogram = main_histogram / np.sum(main_histogram)
		return main_histogram
