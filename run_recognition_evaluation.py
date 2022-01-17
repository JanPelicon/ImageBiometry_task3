import cv2
import glob
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
from feature_extractors.lbp.extractor import LBP
from feature_extractors.hog.extractor import Hog
from feature_extractors.pix2pix.extractor import Pix2Pix


class EvaluateAll:

    def __init__(self):
        self.test_images_path = "data_ears/test"
        self.train_images_path = "data_ears/train"
        self.annotations_path = "data_ears/annotations/recognition/ids.csv"

        self.preprocess = Preprocess()
        self.evaluation = Evaluation()

        self.annotations = self.get_annotations()
        self.images, self.classes = self.get_images(False)

    def get_annotations(self):
        print("Reading annotations...")
        annotations = {}
        with open(self.annotations_path) as file:
            lines = file.readlines()
            for line in lines:
                (key, val) = line.split(',')
                key = key.split(".")[0]
                annotations[key] = int(val)
        return annotations

    def get_images(self, limited=False):
        print("Loading images...")
        images = []
        classes = []
        test_image_list = sorted(glob.glob(self.test_images_path + '/*.png', recursive=True))
        train_image_list = sorted(glob.glob(self.train_images_path + '/*.png', recursive=True))
        image_list = test_image_list + train_image_list
        if limited:
            image_list = test_image_list
        for image_name in image_list:
            image = cv2.imread(image_name)
            image = self.preprocess.pipeline(image=image, size=64)
            images.append(image)
            key = self.get_key_from_path(image_name)
            class_id = self.annotations[key]
            classes.append(class_id)
        return images, classes

    @staticmethod
    def get_key_from_path( path):
        return ((path.split("/")[1]).split(".")[0]).replace("\\", "/")

    @staticmethod
    def distance_matrix(data, metric='jensenshannon'):
        return cdist(data, data, metric)

    def run_evaluation(self):
        print("Running evaluation...")
        print("# of images =", len(self.images))

        pix = Pix2Pix()
        lbp_0 = LBP(points=8, radius=1, size=64, window_stride=8, bins=8)
        lbp_1 = LBP(points=8, radius=1, size=64, window_stride=32, bins=128)
        hog_0 = Hog(cells_block=8, pix_cell=8)
        hog_1 = Hog(cells_block=2, pix_cell=12)

        features_pix = []
        features_pix_s = []

        features_lbp_0 = []
        features_lbp_1 = []
        features_lbp_0s = []
        features_lbp_1s = []

        features_lbp_0h = []
        features_lbp_1h = []
        features_lbp_0hs = []
        features_lbp_1hs = []

        features_hog_0 = []
        features_hog_1 = []
        features_hog_0s = []
        features_hog_1s = []

        for index, image in enumerate(self.images):
            if index % 100 == 0:
                print("{}%".format(int(index/(len(self.images)/100))))

            if index == 0:
                print(pix.extract_pixels(image).shape)
                print(lbp_0.extract_1d(image).shape)
                print(lbp_1.extract_1d(image).shape)
                print(lbp_0.extract_hist(image).shape)
                print(lbp_1.extract_hist(image).shape)
                print(hog_0.extract_hog(image).shape)
                print(hog_1.extract_hog(image).shape)

            sobel = self.preprocess.sobel(image)

            features_pix.append(pix.extract_pixels(image))
            features_pix_s.append(pix.extract_pixels(sobel))

            features_lbp_0.append(lbp_0.extract_hist(image))
            features_lbp_1.append(lbp_1.extract_hist(image))
            features_lbp_0s.append(lbp_0.extract_hist(sobel))
            features_lbp_1s.append(lbp_1.extract_hist(sobel))

            features_lbp_0h.append(lbp_0.extract_1d(image))
            features_lbp_1h.append(lbp_1.extract_1d(image))
            features_lbp_0hs.append(lbp_0.extract_1d(sobel))
            features_lbp_1hs.append(lbp_1.extract_1d(sobel))

            features_hog_0.append(hog_0.extract_hog(image))
            features_hog_1.append(hog_1.extract_hog(image))
            features_hog_0s.append(hog_0.extract_hog(sobel))
            features_hog_1s.append(hog_1.extract_hog(sobel))

        print("Calculating distance matrix for Rank-N...")

        ##### PIX

        distmat_pix = self.distance_matrix(features_pix, "jensenshannon")
        distmat_pix_s = self.distance_matrix(features_pix_s, "jensenshannon")
        rank1_pix = self.evaluation.compute_rank1(distmat_pix, self.classes)
        rank1_pix_s = self.evaluation.compute_rank1(distmat_pix_s, self.classes)
        rank5_pix = self.evaluation.compute_rank5(distmat_pix, self.classes)
        rank5_pix_s = self.evaluation.compute_rank5(distmat_pix_s, self.classes)
        rank10_pix = self.evaluation.compute_rank10(distmat_pix, self.classes)
        rank10_pix_s = self.evaluation.compute_rank10(distmat_pix_s, self.classes)

        print("jensenshannon")
        print("Pixels")
        print("Rank1 = {:.2f}".format(rank1_pix))
        print("Rank5 = {:.2f}".format(rank5_pix))
        print("Rank10 = {:.2f}".format(rank10_pix))
        print("Pixels - SOBEL")
        print("Rank1 = {:.2f}".format(rank1_pix_s))
        print("Rank5 = {:.2f}".format(rank5_pix_s))
        print("Rank10 = {:.2f}".format(rank10_pix_s))
        print()

        distmat_pix = self.distance_matrix(features_pix, "cosine")
        distmat_pix_s = self.distance_matrix(features_pix_s, "cosine")
        rank1_pix = self.evaluation.compute_rank1(distmat_pix, self.classes)
        rank1_pix_s = self.evaluation.compute_rank1(distmat_pix_s, self.classes)
        rank5_pix = self.evaluation.compute_rank5(distmat_pix, self.classes)
        rank5_pix_s = self.evaluation.compute_rank5(distmat_pix_s, self.classes)
        rank10_pix = self.evaluation.compute_rank10(distmat_pix, self.classes)
        rank10_pix_s = self.evaluation.compute_rank10(distmat_pix_s, self.classes)

        print("cosine")
        print("Pixels")
        print("Rank1 = {:.2f}".format(rank1_pix))
        print("Rank5 = {:.2f}".format(rank5_pix))
        print("Rank10 = {:.2f}".format(rank10_pix))
        print("Pixels - SOBEL")
        print("Rank1 = {:.2f}".format(rank1_pix_s))
        print("Rank5 = {:.2f}".format(rank5_pix_s))
        print("Rank10 = {:.2f}".format(rank10_pix_s))
        print()

        ##### LBP_RAW

        distmat_lbp_0h = self.distance_matrix(features_lbp_0h, "jensenshannon")
        distmat_lbp_1h = self.distance_matrix(features_lbp_1h, "jensenshannon")
        rank1_lbp_0h = self.evaluation.compute_rank1(distmat_lbp_0h, self.classes)
        rank1_lbp_1h = self.evaluation.compute_rank1(distmat_lbp_1h, self.classes)
        rank5_lbp_0h = self.evaluation.compute_rank5(distmat_lbp_0h, self.classes)
        rank5_lbp_1h = self.evaluation.compute_rank5(distmat_lbp_1h, self.classes)
        rank10_lbp_0h = self.evaluation.compute_rank10(distmat_lbp_0h, self.classes)
        rank10_lbp_1h = self.evaluation.compute_rank10(distmat_lbp_1h, self.classes)

        print("jensenshannon")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0h))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0h))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0h))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1h))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1h))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1h))
        print()

        distmat_lbp_0h = self.distance_matrix(features_lbp_0h, "cosine")
        distmat_lbp_1h = self.distance_matrix(features_lbp_1h, "cosine")
        rank1_lbp_0h = self.evaluation.compute_rank1(distmat_lbp_0h, self.classes)
        rank1_lbp_1h = self.evaluation.compute_rank1(distmat_lbp_1h, self.classes)
        rank5_lbp_0h = self.evaluation.compute_rank5(distmat_lbp_0h, self.classes)
        rank5_lbp_1h = self.evaluation.compute_rank5(distmat_lbp_1h, self.classes)
        rank10_lbp_0h = self.evaluation.compute_rank10(distmat_lbp_0h, self.classes)
        rank10_lbp_1h = self.evaluation.compute_rank10(distmat_lbp_1h, self.classes)

        print("cosine")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0h))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0h))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0h))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1h))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1h))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1h))
        print()

        ##### LBP_RAW SOBEL

        distmat_lbp_0hs = self.distance_matrix(features_lbp_0hs, "jensenshannon")
        distmat_lbp_1hs = self.distance_matrix(features_lbp_1hs, "jensenshannon")
        rank1_lbp_0hs = self.evaluation.compute_rank1(distmat_lbp_0hs, self.classes)
        rank1_lbp_1hs = self.evaluation.compute_rank1(distmat_lbp_1hs, self.classes)
        rank5_lbp_0hs = self.evaluation.compute_rank5(distmat_lbp_0hs, self.classes)
        rank5_lbp_1hs = self.evaluation.compute_rank5(distmat_lbp_1hs, self.classes)
        rank10_lbp_0hs = self.evaluation.compute_rank10(distmat_lbp_0hs, self.classes)
        rank10_lbp_1hs = self.evaluation.compute_rank10(distmat_lbp_1hs, self.classes)

        print("SOBEL - jensenshannon")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0hs))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0hs))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0hs))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1hs))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1hs))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1hs))
        print()

        distmat_lbp_0hs = self.distance_matrix(features_lbp_0hs, "cosine")
        distmat_lbp_1hs = self.distance_matrix(features_lbp_1hs, "cosine")
        rank1_lbp_0hs = self.evaluation.compute_rank1(distmat_lbp_0hs, self.classes)
        rank1_lbp_1hs = self.evaluation.compute_rank1(distmat_lbp_1hs, self.classes)
        rank5_lbp_0hs = self.evaluation.compute_rank5(distmat_lbp_0hs, self.classes)
        rank5_lbp_1hs = self.evaluation.compute_rank5(distmat_lbp_1hs, self.classes)
        rank10_lbp_0hs = self.evaluation.compute_rank10(distmat_lbp_0hs, self.classes)
        rank10_lbp_1hs = self.evaluation.compute_rank10(distmat_lbp_1hs, self.classes)

        print("SOBEL - cosine")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0hs))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0hs))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0hs))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1hs))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1hs))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1hs))
        print()

        ##### LBP HIST

        distmat_lbp_0 = self.distance_matrix(features_lbp_0, "jensenshannon")
        distmat_lbp_1 = self.distance_matrix(features_lbp_1, "jensenshannon")
        rank1_lbp_0 = self.evaluation.compute_rank1(distmat_lbp_0, self.classes)
        rank1_lbp_1 = self.evaluation.compute_rank1(distmat_lbp_1, self.classes)
        rank5_lbp_0 = self.evaluation.compute_rank5(distmat_lbp_0, self.classes)
        rank5_lbp_1 = self.evaluation.compute_rank5(distmat_lbp_1, self.classes)
        rank10_lbp_0 = self.evaluation.compute_rank10(distmat_lbp_0, self.classes)
        rank10_lbp_1 = self.evaluation.compute_rank10(distmat_lbp_1, self.classes)

        print("jensenshannon")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1))
        print()

        distmat_lbp_0 = self.distance_matrix(features_lbp_0, "cosine")
        distmat_lbp_1 = self.distance_matrix(features_lbp_1, "cosine")
        rank1_lbp_0 = self.evaluation.compute_rank1(distmat_lbp_0, self.classes)
        rank1_lbp_1 = self.evaluation.compute_rank1(distmat_lbp_1, self.classes)
        rank5_lbp_0 = self.evaluation.compute_rank5(distmat_lbp_0, self.classes)
        rank5_lbp_1 = self.evaluation.compute_rank5(distmat_lbp_1, self.classes)
        rank10_lbp_0 = self.evaluation.compute_rank10(distmat_lbp_0, self.classes)
        rank10_lbp_1 = self.evaluation.compute_rank10(distmat_lbp_1, self.classes)

        print("cosine")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1))
        print()

        ##### LBP HIST SOBEL

        distmat_lbp_0s = self.distance_matrix(features_lbp_0s, "jensenshannon")
        distmat_lbp_1s = self.distance_matrix(features_lbp_1s, "jensenshannon")
        rank1_lbp_0 = self.evaluation.compute_rank1(distmat_lbp_0s, self.classes)
        rank1_lbp_1 = self.evaluation.compute_rank1(distmat_lbp_1s, self.classes)
        rank5_lbp_0 = self.evaluation.compute_rank5(distmat_lbp_0s, self.classes)
        rank5_lbp_1 = self.evaluation.compute_rank5(distmat_lbp_1s, self.classes)
        rank10_lbp_0 = self.evaluation.compute_rank10(distmat_lbp_0s, self.classes)
        rank10_lbp_1 = self.evaluation.compute_rank10(distmat_lbp_1s, self.classes)

        print("SOBEL - jensenshannon")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1))
        print()

        distmat_lbp_0s = self.distance_matrix(features_lbp_0s, "cosine")
        distmat_lbp_1s = self.distance_matrix(features_lbp_1s, "cosine")
        rank1_lbp_0 = self.evaluation.compute_rank1(distmat_lbp_0s, self.classes)
        rank1_lbp_1 = self.evaluation.compute_rank1(distmat_lbp_1s, self.classes)
        rank5_lbp_0 = self.evaluation.compute_rank5(distmat_lbp_0s, self.classes)
        rank5_lbp_1 = self.evaluation.compute_rank5(distmat_lbp_1s, self.classes)
        rank10_lbp_0 = self.evaluation.compute_rank10(distmat_lbp_0s, self.classes)
        rank10_lbp_1 = self.evaluation.compute_rank10(distmat_lbp_1s, self.classes)

        print("SOBEL - cosine")
        print("LBP(points=8, radius=1, size=64, window_stride=8, bins=8)")
        print("Rank1 LBP_0 = {:.2f}".format(rank1_lbp_0))
        print("Rank5 LBP_0 = {:.2f}".format(rank5_lbp_0))
        print("Rank10 LBP_0 = {:.2f}".format(rank10_lbp_0))
        print("LBP(points=8, radius=1, size=64, window_stride=32, bins=128)")
        print("Rank1 LBP_1 = {:.2f}".format(rank1_lbp_1))
        print("Rank5 LBP_1 = {:.2f}".format(rank5_lbp_1))
        print("Rank10 LBP_1 = {:.2f}".format(rank10_lbp_1))
        print()

        ##### HOG

        distmat_hog_0 = self.distance_matrix(features_hog_0, "jensenshannon")
        distmat_hog_1 = self.distance_matrix(features_hog_1, "jensenshannon")
        rank1_hog_0 = self.evaluation.compute_rank1(distmat_hog_0, self.classes)
        rank1_hog_1 = self.evaluation.compute_rank1(distmat_hog_1, self.classes)
        rank5_hog_0 = self.evaluation.compute_rank5(distmat_hog_0, self.classes)
        rank5_hog_1 = self.evaluation.compute_rank5(distmat_hog_1, self.classes)
        rank10_hog_0 = self.evaluation.compute_rank10(distmat_hog_0, self.classes)
        rank10_hog_1 = self.evaluation.compute_rank10(distmat_hog_1, self.classes)

        print("jensenshannon")
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_0))
        print("Rank5 = {:.2f}".format(rank5_hog_0))
        print("Rank10 = {:.2f}".format(rank10_hog_0))
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_1))
        print("Rank5 = {:.2f}".format(rank5_hog_1))
        print("Rank10 = {:.2f}".format(rank10_hog_1))
        print()

        distmat_hog_0 = self.distance_matrix(features_hog_0, "cosine")
        distmat_hog_1 = self.distance_matrix(features_hog_1, "cosine")
        rank1_hog_0 = self.evaluation.compute_rank1(distmat_hog_0, self.classes)
        rank1_hog_1 = self.evaluation.compute_rank1(distmat_hog_1, self.classes)
        rank5_hog_0 = self.evaluation.compute_rank5(distmat_hog_0, self.classes)
        rank5_hog_1 = self.evaluation.compute_rank5(distmat_hog_1, self.classes)
        rank10_hog_0 = self.evaluation.compute_rank10(distmat_hog_0, self.classes)
        rank10_hog_1 = self.evaluation.compute_rank10(distmat_hog_1, self.classes)


        print("cosine")
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_0))
        print("Rank5 = {:.2f}".format(rank5_hog_0))
        print("Rank10 = {:.2f}".format(rank10_hog_0))
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_1))
        print("Rank5 = {:.2f}".format(rank5_hog_1))
        print("Rank10 = {:.2f}".format(rank10_hog_1))
        print()

        ##### HOG SOBEL

        distmat_hog_0s = self.distance_matrix(features_hog_0s, "jensenshannon")
        distmat_hog_1s = self.distance_matrix(features_hog_1s, "jensenshannon")
        rank1_hog_0s = self.evaluation.compute_rank1(distmat_hog_0s, self.classes)
        rank1_hog_1s = self.evaluation.compute_rank1(distmat_hog_1s, self.classes)
        rank5_hog_0s = self.evaluation.compute_rank5(distmat_hog_0s, self.classes)
        rank5_hog_1s = self.evaluation.compute_rank5(distmat_hog_1s, self.classes)
        rank10_hog_0s = self.evaluation.compute_rank10(distmat_hog_0s, self.classes)
        rank10_hog_1s = self.evaluation.compute_rank10(distmat_hog_1s, self.classes)

        print("jensenshannon - SOBEL")
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_0s))
        print("Rank5 = {:.2f}".format(rank5_hog_0s))
        print("Rank10 = {:.2f}".format(rank10_hog_0s))
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_1s))
        print("Rank5 = {:.2f}".format(rank5_hog_1s))
        print("Rank10 = {:.2f}".format(rank10_hog_1s))
        print()

        distmat_hog_0s = self.distance_matrix(features_hog_0s, "cosine")
        distmat_hog_1s = self.distance_matrix(features_hog_1s, "cosine")
        rank1_hog_0s = self.evaluation.compute_rank1(distmat_hog_0s, self.classes)
        rank1_hog_1s = self.evaluation.compute_rank1(distmat_hog_1s, self.classes)
        rank5_hog_0s = self.evaluation.compute_rank5(distmat_hog_0s, self.classes)
        rank5_hog_1s = self.evaluation.compute_rank5(distmat_hog_1s, self.classes)
        rank10_hog_0s = self.evaluation.compute_rank10(distmat_hog_0s, self.classes)
        rank10_hog_1s = self.evaluation.compute_rank10(distmat_hog_1s, self.classes)

        print("cosine - SOBEL")
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_0s))
        print("Rank5 = {:.2f}".format(rank5_hog_0s))
        print("Rank10 = {:.2f}".format(rank10_hog_0s))
        print("Hog")
        print("Rank1 = {:.2f}".format(rank1_hog_1s))
        print("Rank5 = {:.2f}".format(rank5_hog_1s))
        print("Rank10 = {:.2f}".format(rank10_hog_1s))
        print()

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
