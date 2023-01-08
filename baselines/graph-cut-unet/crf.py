# code modified from: https://github.com/dhawan98/Post-Processing-of-Image-Segmentation-using-CRF/blob/master/CRF%20initial(running).ipynb

import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import cv2

from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from osgeo import gdal


def crf(original_image, annotated_image, output_image):
    if len(annotated_image.shape) < 3:
        annotated_image = gray2rgb(annotated_image).astype(np.uint32)

    cv2.imwrite("testing2.png", annotated_image)
    annotated_image = annotated_image.astype(np.uint32)

    annotated_label = annotated_image[:, :, 0].astype(np.uint32) + (annotated_image[:, :, 1] << 8).astype(np.uint32) + (
                annotated_image[:, :, 2] << 16).astype(np.uint32)

    colors, labels = np.unique(annotated_label, return_inverse=True)

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat))

    print("No of labels in the Image are ")
    print(n_labels)


    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.90, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                            compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)

    MAP = np.argmax(Q, axis=0)

    MAP = colorize[MAP, :]
    cv2.imwrite(output_image, MAP.reshape(original_image.shape))
    return MAP.reshape(original_image.shape)