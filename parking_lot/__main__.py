#!/usr/bin/python

import glob

import cv2 as cv
import numpy as np
import click
import xgboost as xgb
import sklearn.neural_network as sknn

import util
import conf
from combined_signaller import CombinedSignaller
from cnn import CNNSignaller


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
        signaller: CombinedSignaller,
        classifier: sknn.MLPClassifier,
        pkm_coordinates: list[list]) -> tuple[int, int]:

    img = cv.imread(img_name, 0)
    img_cpy = img.copy()

    correct_results = load_results_for_img(img_name)
    res = []

    for coord in pkm_coordinates:
        place = extract_parking_space(img, coord)

        signals = signaller.get_signals(place)

        # mlp_pred = classifier.predict(np.asarray([signals]))

        # occupied = mlp_pred[0]
        occupied = signals

        res.append(occupied)
        if occupied:
            mark_occupied(coord, img_cpy)

    succ, total = util.cmp_results(res, correct_results)

    cv.imshow('Zpiceny eduroam', img_cpy)
    cv.waitKey(0)

    return succ, total


@click.command()
@click.option('--lbp-model', help='Path to XGB model used in prediction')
@click.option('--hog-model', help='Path to SVM model used in prediction')
@click.option('--cnn-model', help='Path to CNN model used in prediction')
@click.option('--final-classifier-model', help='Path to MLP model')
def classify(lbp_model, hog_model, cnn_model, final_classifier_model) -> None:

    pkm_coordinates = load_coords('data/parking_map_python.txt')
    test_images = sorted([img for img in glob.glob('data/test_images/*.jpg')])
    lbp_booster = util.load_booster(lbp_model)
    hog_svm = cv.ml.SVM.load(hog_model)
    cnn = CNNSignaller(cnn_model)

    signaller = CombinedSignaller(lbp=lbp_booster, hog=hog_svm, cnn=cnn)
    classifier = util.load_final_classifier(final_classifier_model)

    total_successful = 0
    total = 0

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        s, t = process_img(
                img_name=img_name,
                signaller=signaller,
                classifier=classifier,
                pkm_coordinates=pkm_coordinates)
        total_successful += s
        total += t

    print(f'Total success rate: {total_successful / total}')


@click.group('Classification')
def main() -> None:
    pass


# TODO: Use LBP
main.add_command(classify)


if __name__ == "__main__":
    main()
