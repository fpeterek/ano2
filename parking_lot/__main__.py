#!/usr/bin/python

import glob

import cv2 as cv
import numpy as np
import click
import xgboost as xgb

import util
import conf
from edge_predictor import EdgePredictor


def load_coords(filename: str):
    pkm_coordinates = []
    with open(filename) as pkm_file:
        pkm_lines = pkm_file.readlines()

        for line in pkm_lines:
            st_line = line.strip()
            sp_line = list(st_line.split(" "))
            pkm_coordinates.append(sp_line)

    return pkm_coordinates


def mark_occupied(coordinates: list, target_img) -> None:
    p1 = int(coordinates[0]), int(coordinates[1]),
    p2 = int(coordinates[2]), int(coordinates[3]),
    p3 = int(coordinates[4]), int(coordinates[5]),
    p4 = int(coordinates[6]), int(coordinates[7]),

    cv.line(target_img, p1, p3, 255, 2)
    cv.line(target_img, p2, p4, 255, 2)


def extract_parking_space(orig_img, coordinates):
    place = util.four_point_transform(orig_img, coordinates)
    place = cv.medianBlur(place, 3)
    return cv.resize(place, conf.img_dim)


def load_results_for_img(img_name: str) -> list[bool]:
    res_file = f'{img_name[:-3]}txt'
    return util.load_res(res_file)


def lbp_prediction(lbp, booster, img):
    hist = xgb.DMatrix(np.matrix(lbp(img)))
    return booster.predict(hist)[0]


def hog_prediction(hog, svm, img):
    hog_sigs = hog.compute(img)
    return svm.predict(np.matrix(hog_sigs))[1][0][0]


def process_img(
        img_name: str,
        lbp_booster: xgb.Booster,
        hog_model,
        edge_predictor: EdgePredictor,
        pkm_coordinates: list[list]):

    img = cv.imread(img_name, 0)
    img_cpy = img.copy()

    correct_results = load_results_for_img(img_name)
    res = []

    lbp = util.create_lbp_signaller()
    hog = util.create_hog_descriptor()

    for coord in pkm_coordinates:
        place = extract_parking_space(img, coord)

        lbp_pred = lbp_prediction(lbp, lbp_booster, place)
        hog_pred = hog_prediction(hog, hog_model, place)
        edge_pred = edge_predictor.predict(place)

        # occupied = lbp_pred > 0.8 and hog_pred > 0.5 and edge_pred > 0.35
        weights = [1.0, 1.0, 2.0]
        preds = [lbp_pred, hog_pred, edge_pred]

        occupied = sum([p*w for p, w in zip(preds, weights)]) / sum(weights)
        occupied = occupied > 0.5

        res.append(occupied)
        if occupied:
            mark_occupied(coord, img_cpy)

    util.cmp_results(res, correct_results)
    cv.imshow('Zpiceny eduroam', img_cpy)
    cv.waitKey(0)


@click.command()
@click.option('--lbp-model', help='Path to XGB model used in prediction')
@click.option('--hog-model', help='Path to SVM model used in prediction')
def classify(lbp_model: str, hog_model: str) -> None:

    pkm_coordinates = load_coords('data/parking_map_python.txt')
    test_images = sorted([img for img in glob.glob('data/test_images/*.jpg')])
    lbp_booster = util.load_booster(lbp_model)
    hog_svm = cv.ml.SVM.load(hog_model)
    edge_pred = EdgePredictor()

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        process_img(
                img_name=img_name,
                lbp_booster=lbp_booster,
                hog_model=hog_svm,
                edge_predictor=edge_pred,
                pkm_coordinates=pkm_coordinates)


@click.group('Classification')
def main() -> None:
    pass


# TODO: Use LBP
main.add_command(classify)


if __name__ == "__main__":
    main()
