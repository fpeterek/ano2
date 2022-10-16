#!/usr/bin/python

import glob

import cv2 as cv
import numpy as np
import click
import xgboost as xgb

import util
import conf


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

    is_car_ratio = 0.030

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        img = cv.imread(img_name, 0)
        img_cpy = img.copy()

        res_file = f'{img_name[:-3]}txt'
        correct_results = util.load_res(res_file)
        res = []

        for coord in pkm_coordinates:
            place = util.four_point_transform(img, coord)
            place = cv.medianBlur(place, 3)

            des_height, des_width = conf.img_dim

            place = cv.resize(place, (des_height, des_width))

            canny_img = cv.Canny(place, 150, 200)
            # cv.imshow('Canny', canny_img)
            # cv.waitKey(0)

            non_zero_ratio = 0

            for y in range(canny_img.shape[0]):
                for x in range(canny_img.shape[1]):
                    non_zero_ratio += canny_img[y, x] > 127

            non_zero_ratio /= des_width * des_height

            p1 = int(coord[0]), int(coord[1]),
            p2 = int(coord[2]), int(coord[3]),
            p3 = int(coord[4]), int(coord[5]),
            p4 = int(coord[6]), int(coord[7]),

            res.append(non_zero_ratio >= is_car_ratio)
            if non_zero_ratio >= is_car_ratio:
                cv.line(img_cpy, p1, p3, 255, 2)
                cv.line(img_cpy, p2, p4, 255, 2)

        util.cmp_results(res, correct_results)
        cv.imshow('Zpiceny eduroam', img_cpy)
        cv.waitKey(0)


@click.command()
@click.option('--model-name', help='Name of model')
@click.option('--enable-edge-detection', default=False,
              help='Enable edge detection')
def hog_classifier(model_name: str, enable_edge_detection: bool):
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()

    hog = util.create_hog_descriptor()
    svm = cv.ml.SVM.load(model_name)

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        img = cv.imread(img_name, 0)
        img_cpy = img.copy()

        res_file = f'{img_name[:-3]}txt'
        correct_results = util.load_res(res_file)
        res = []

        edge_and_hog_ratio = 0.02
        edge_only_ratio = 0.07

        for coord in pkm_coordinates:
            place = util.four_point_transform(img, coord)
            place = cv.medianBlur(place, 3)

            place = cv.resize(place, conf.img_dim)

            hog_sigs = hog.compute(place)
            occupied = svm.predict(np.matrix(hog_sigs))[1][0] > 0.5

            if enable_edge_detection:
                canny_img = cv.Canny(place, 150, 200)

                non_zero_ratio = 0

                for y in range(canny_img.shape[0]):
                    for x in range(canny_img.shape[1]):
                        non_zero_ratio += canny_img[y, x] > 127

                non_zero_ratio /= conf.img_dim[0] * conf.img_dim[1]

                occupied = \
                    (occupied and non_zero_ratio > edge_and_hog_ratio) or \
                    non_zero_ratio > edge_only_ratio

            p1 = int(coord[0]), int(coord[1]),
            p2 = int(coord[2]), int(coord[3]),
            p3 = int(coord[4]), int(coord[5]),
            p4 = int(coord[6]), int(coord[7]),

            res.append(occupied)
            if occupied:
                cv.line(img_cpy, p1, p3, 255, 2)
                cv.line(img_cpy, p2, p4, 255, 2)

        util.cmp_results(res, correct_results)
        cv.imshow('Zpiceny eduroam', img_cpy)
        cv.waitKey(0)


@click.command()
@click.option('--xgb-model', help='Path to XGB model used in prediction')
def lbp_classifier(xgb_model: str) -> None:
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images/*.jpg")]

    lbp = util.create_lbp_signaller()
    booster = xgb.Booster()
    booster.load_model(xgb_model)

    test_images.sort()

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        img = cv.imread(img_name, 0)
        img_cpy = img.copy()

        res_file = f'{img_name[:-3]}txt'
        correct_results = util.load_res(res_file)
        res = []

        for coord in pkm_coordinates:
            place = util.four_point_transform(img, coord)
            place = cv.medianBlur(place, 3)

            place = cv.resize(place, conf.img_dim)

            hist = xgb.DMatrix(np.matrix(lbp(place)))
            occupied = booster.predict(hist)[0] >= 0.8

            p1 = int(coord[0]), int(coord[1]),
            p2 = int(coord[2]), int(coord[3]),
            p3 = int(coord[4]), int(coord[5]),
            p4 = int(coord[6]), int(coord[7]),

            res.append(occupied)
            if occupied:
                cv.line(img_cpy, p1, p3, 255, 2)
                cv.line(img_cpy, p2, p4, 255, 2)

        util.cmp_results(res, correct_results)
        cv.imshow('Zpiceny eduroam', img_cpy)
        cv.waitKey(0)


@click.group('Classification')
def main() -> None:
    pass


# TODO: Use LBP

main.add_command(default_classifier)
main.add_command(hog_classifier)
main.add_command(lbp_classifier)


if __name__ == "__main__":
    main()
