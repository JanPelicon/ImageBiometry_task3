from skimage.feature import hog


class Hog:
	def __init__(self, cells_block=8, pix_cell=3):
		self.cells_block = (cells_block, cells_block)
		self.pix_cell = (pix_cell, pix_cell)

	def extract_hog(self, image):
		feature = hog(image, orientations=8, pixels_per_cell=self.pix_cell, cells_per_block=self.cells_block)
		return feature
