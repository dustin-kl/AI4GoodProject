import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from tkinter import *
from PIL import Image
import os

from graph_cut import GraphCut


class GraphCutController:
    def __init__(self, image, seed_fg, seed_bg, lambda_value, image_path):
        self.image_path = image_path
        self.image = image
        self.seed_fg = seed_fg
        self.seed_bg = seed_bg
        self.lambda_value = lambda_value

        self.segment_image()

    def __get_color_histogram(self, image, seed, hist_res):
        seed_r = image[seed[:, 1], seed[:, 0], 0]
        seed_g = image[seed[:, 1], seed[:, 0], 1]
        seed_b = image[seed[:, 1], seed[:, 0], 2]

        seed_rgb = np.vstack((seed_r, seed_g, seed_b))
        rgb = np.transpose(seed_rgb)
        hist, _ = np.histogramdd(rgb, hist_res, range=[(0, 255), (0, 255), (0, 255)])

        hist_smooth = ndimage.gaussian_filter(hist, 0.1)
        hist_smooth /= np.sum(hist_smooth.ravel())

        return hist_smooth

    def __get_unaries(self, image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
        H = image.shape[0]
        W = image.shape[1]
        U = np.zeros((H, W, 2))
        eps = 1e-10
        for i in range(H):
            for j in range(W):
                pixel = image[i, j, :]
                bins = np.floor(pixel / (255./32.)).astype(int)
                bins[bins == 32] = 31

                cost_fg = -np.log(hist_fg[bins[0], bins[1], bins[2]] + eps)
                cost_bg = -np.log(hist_bg[bins[0], bins[1], bins[2]] + eps)

                U[i, j, 0] = cost_fg * lambda_param
                U[i, j, 1] = cost_bg * lambda_param

        for j, i in seed_fg:
            U[i, j, 0] = 0
            U[i, j, 1] = np.inf

        for j, i in seed_bg:
            U[i, j, 0] = np.inf
            U[i, j, 1] = 0

        return np.reshape(U, (-1, 2))

    def __get_neighbours(self, i, j, H, W):
        neighbours = []
        for h in (-1, 0, 1):
            for w in (-1, 0, 1):
                if h == 0 and w == 0:
                    continue
                x = i + h
                y = j + w
                if 0 <= x < H and 0 <= y < W:
                    neighbours.append([x, y])

        return np.array(neighbours)

    def __get_pairwise(self, image):
        H = image.shape[0]
        W = image.shape[1]
        P = []
        sigma = 5
        for i in range(H):
            for j in range(W):
                pos = np.array([i, j])
                idx = i * W + j
                pixel = image[i, j].astype(float)
                neighbours = self.__get_neighbours(i, j, H, W)
                neighbours_indices = neighbours[:, 0] * W + neighbours[:, 1]
                neighbours_pixels = image[neighbours[:, 0], neighbours[:, 1]].astype(float)

                pixel_diff = neighbours_pixels - pixel
                pixel_dist = np.linalg.norm(pixel_diff, axis=1)
                spatial_diff = pos - neighbours
                spatial_dist = np.linalg.norm(spatial_diff, axis=1)

                costs = np.divide(np.exp(- pixel_dist ** 2 / (2 * sigma ** 2)), spatial_dist + 1e-10)

                num_neighbours = neighbours_indices.shape[0]
                for k in range(num_neighbours):
                    n_idx = neighbours_indices[k]
                    cost = costs[k]
                    P.append([idx, n_idx, 0, cost, 0, 0])

        return np.asarray(P)

    def __get_segmented_image(self, image, labels, background=None):
        H = image.shape[0]
        W = image.shape[1]

        not_indices = np.logical_not(labels)
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        mask[not_indices, :] = np.array([255, 255, 255], dtype=np.uint8)
        mask[labels, :] = np.array([0, 0, 0], dtype=np.uint8)

        segmented = mask

        res = None
        if background is not None:
            mask = np.zeros((H, W), dtype=np.bool)
            mask[not_indices] = True
            res = np.copy(background[0:H, 0:W, :])
            res.setflags(write=True)
            res[not_indices, 0:3] = image[not_indices, 0:3]

        return segmented, res

    def segment_image(self, background=None):
        image = self.image
        seed_fg = self.seed_fg
        seed_bg = self.seed_bg
        lambda_value = self.lambda_value

        image = image.convert("RGB")
        image_array = np.asarray(image)
        background_array = None
        if background:
            background = background.convert("RGB")
            background_array = np.asarray(background)
        seed_fg = np.array(seed_fg)
        seed_bg = np.array(seed_bg)
        height, width = np.shape(image_array)[0:2]
        num_pixels = height * width

        hist_res = 32
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array)

        graph = GraphCut(num_pixels, pairwise.__len__())
        graph.set_unary(unaries)
        graph.set_pairwise(pairwise)
        graph.minimize()
        labels = graph.get_labeling()
        labels = np.reshape(labels, (height, width))

        segmented_image, segmented_image_with_background = self.__get_segmented_image(image_array, labels, background_array)
        segmented_image = Image.fromarray(segmented_image, 'RGB')

        output_path = self.image_path.replace("images", "output")
        segmented_image.save(output_path)

        if segmented_image_with_background is not None:
            segmented_image_with_background = Image.fromarray(segmented_image_with_background, 'RGB')
            plt.imshow(segmented_image_with_background)
            plt.show()
