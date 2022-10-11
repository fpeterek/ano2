#!/usr/bin/python

import sys
import glob
import random

import cv2 as cv
import numpy as np
import click


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


@click.command()
def default_classifier():

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    print(pkm_coordinates)
    print("********************************************************")

    is_car_ratio = 0.030

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        print(img_name)
        img = cv.imread(img_name, 0)
        img_cpy = img.copy()

        res_file = f'{img_name[:-3]}txt'
        correct_results = load_res(res_file)
        res = []

        for coord in pkm_coordinates:
            place = four_point_transform(img, coord)
            place = cv.medianBlur(place, 3)

            height, width = place.shape
            des_height = 64
            des_width = round((height / des_height) * width)

            place = cv.resize(place, (des_height, des_width))

            canny_img = cv.Canny(place, 150, 200)
            # cv.imshow('Canny', canny_img)
            # cv.waitKey(0)

            non_zero_ratio = 0

            for y in range(canny_img.shape[0]):
                for x in range(canny_img.shape[1]):
                    non_zero_ratio += canny_img[y, x] > 127

            non_zero_ratio /= des_width * des_height
            print(non_zero_ratio)

            p1 = int(coord[0]), int(coord[1]),
            p2 = int(coord[2]), int(coord[3]),
            p3 = int(coord[4]), int(coord[5]),
            p4 = int(coord[6]), int(coord[7]),

            res.append(non_zero_ratio >= is_car_ratio)
            if non_zero_ratio >= is_car_ratio:
                cv.line(img_cpy, p1, p3, 255, 2)
                cv.line(img_cpy, p2, p4, 255, 2)

        cmp_results(res, correct_results)
        cv.imshow('Zpiceny eduroam', img_cpy)
        cv.waitKey(0)


def create_hog_descriptor() -> cv.HOGDescriptor:
    win_size = (20, 20)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (10, 10)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1.
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    signed_gradients = True

    return cv.HOGDescriptor(
            win_size,
            block_size,
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


def load_ds(hog: cv.HOGDescriptor, folder: str, label: bool | int) -> list[tuple]:
    label = int(label)

    signals = []

    for img_name in glob.glob(f'{folder}/*'):
        img = cv.imread(img_name, 0)
        img = cv.medianBlur(img, 3)
        img = cv.resize(img, (128, 64))

        hog_sigs = hog.compute(img)

        signals.append((hog_sigs, label))

    return signals


@click.command()
@click.option('--free-set', help='Folder with images of free parking spaces')
@click.option('--occupied-set',
              help='Folder with images of occupied parking spaces')
@click.option('--model-name', help='Name of model')
def train(free_set: str, occupied_set: str, model_name: str):
    hog = create_hog_descriptor()

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_RBF)
    svm.setC(12.5)
    svm.setGamma(0.5)

    # TODO: train
    s1 = load_ds(hog, free_set, 0)
    s2 = load_ds(hog, occupied_set, 1)

    sigs = s1 + s2
    random.shuffle(sigs)

    signals = np.matrix([s for s, l in sigs])
    labels = np.array([l for s, l in sigs])

    svm.train(signals, cv.ml.ROW_SAMPLE, labels)

    svm.save(model_name)


@click.command()
@click.option('--model-name', help='Name of model')
def hog_classifier(model_name: str):
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    print(pkm_coordinates)
    print("********************************************************")

    hog = hog_classifier()
    svm = cv.ml.SVM.load(model_name)

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        print(img_name)
        img = cv.imread(img_name, 0)
        img_cpy = img.copy()

        res_file = f'{img_name[:-3]}txt'
        correct_results = load_res(res_file)
        res = []

        for coord in pkm_coordinates:
            place = four_point_transform(img, coord)
            place = cv.medianBlur(place, 3)

            height, width = place.shape
            des_height = 64
            des_width = round((height / des_height) * width)

            place = cv.resize(place, (des_height, des_width))

            hog_sigs = hog.compute(place)
            occupied, _ = svm.predict(hog_sigs)

            # occupied = 0

            p1 = int(coord[0]), int(coord[1]),
            p2 = int(coord[2]), int(coord[3]),
            p3 = int(coord[4]), int(coord[5]),
            p4 = int(coord[6]), int(coord[7]),

            res.append()
            if occupied:
                cv.line(img_cpy, p1, p3, 255, 2)
                cv.line(img_cpy, p2, p4, 255, 2)

        cmp_results(res, correct_results)
        cv.imshow('Zpiceny eduroam', img_cpy)
        cv.waitKey(0)
    pass


@click.group('ANO 2')
def main() -> None:
    pass


main.add_command(default_classifier)


if __name__ == "__main__":
    main()
