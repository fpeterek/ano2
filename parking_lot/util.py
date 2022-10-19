import glob
import random
import pickle

import numpy as np
import cv2 as cv
import skimage.feature as skf
import xgboost as xgb
import sklearn.neural_network as sknn

import conf


def four_point_transform(image, one_c):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
       [0, 0],
       [maxWidth - 1, 0],
       [maxWidth - 1, maxHeight - 1],
       [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
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


def load_res(filename: str) -> list[bool]:
    res = []
    with open(filename) as file:
        for line in file:
            if line:
                res.append('1' in line)
    return res


def cmp_results(res: list[bool], correct: list[bool]) -> None:
    count = 0

    for rec, exp in zip(res, correct):
        count += rec == exp

    print(f'Success rate: {count / len(res)}')


def create_hog_descriptor() -> cv.HOGDescriptor:
    win_size = (96, 96)
    block_size = (32, 32)
    block_stride = (16, 16)
    cell_size = (8, 8)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    signed_gradients = True

    return cv.HOGDescriptor(
            win_size,
            block_size,
            block_stride,
            cell_size,
            nbins,
            deriv_aperture,
            win_sigma,
            histogram_norm_type,
            l2_hys_threshold,
            gamma_correction,
            nlevels,
            signed_gradients)


def create_lbp_signaller():
    radius = 3
    n_points = 8 * radius
    method = 'uniform'

    def signaller(img):
        lbp = skf.local_binary_pattern(img, radius, n_points, method)

        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist

    return signaller


def load_folder(signaller, folder: str, label: bool | int) -> list[tuple]:
    label = int(label)

    signals = []

    for img_name in glob.glob(f'{folder}/*'):
        img = cv.imread(img_name, 0)
        img = cv.medianBlur(img, 3)
        img = cv.resize(img, conf.img_dim)

        sigs = signaller(img)

        signals.append((sigs, label))

    return signals


def load_training_ds(signaller, free_folder: str, occupied_folder: str):
    s1 = load_folder(signaller, free_folder, 0)
    s2 = load_folder(signaller, occupied_folder, 1)

    sigs = s1 + s2
    random.shuffle(sigs)

    signals = [s for s, l in sigs]
    labels = [l for s, l in sigs]

    return signals, labels


def load_booster(model_name: str) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(model_name)
    return bst


def load_final_classifier(model_name: str) -> sknn.MLPClassifier:
    with open(model_name, 'rb') as file:
        return pickle.load(file)
