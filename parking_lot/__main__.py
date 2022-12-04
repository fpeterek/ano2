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
from cnn import CNNSignaller, ResnetSignaller
from edge_predictor import EdgePredictor
from rcnn import rcnn as rcnn_detection


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
        classifier: sknn.MLPClassifier | None,
        pkm_coordinates: list[list]) -> tuple[int, int]:

    img = cv.imread(img_name, 0)
    img_cpy = img.copy()

    correct_results = load_results_for_img(img_name)
    res = []

    for coord in pkm_coordinates:
        place = extract_parking_space(img, coord)

        signals = signaller(place)

        if classifier:
            mlp_pred = classifier.predict(np.asarray([signals]))
            occupied = mlp_pred[0]
        else:
            occupied = signals

        res.append(occupied)
        if occupied:
            util.mark_occupied(coord, img_cpy)

    succ, total = util.cmp_results(res, correct_results)

    cv.imshow('Zpiceny eduroam', img_cpy)
    cv.waitKey(0)

    return succ, total


@click.command()
@click.option('--cnn-model', help='Path to CNN model used in prediction')
def cnn_classify(cnn_model):
    pkm_coordinates = util.load_coords('data/parking_map_python.txt')
    test_images = sorted([img for img in glob.glob('data/test_images/*.jpg')])
    cnn = CNNSignaller(cnn_model)

    total_successful = 0
    total = 0

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        s, t = process_img(
                img_name=img_name,
                signaller=lambda pic: cnn(pic)[0],
                classifier=None,
                pkm_coordinates=pkm_coordinates)
        total_successful += s
        total += t

    print(f'Total success rate: {total_successful / total}')


@click.command()
@click.option('--resnet-model', help='Path to Resnet model used in prediction')
def resnet_classify(resnet_model):
    pkm_coordinates = util.load_coords('data/parking_map_python.txt')
    test_images = sorted([img for img in glob.glob('data/test_images/*.jpg')])
    cnn = ResnetSignaller(resnet_model)

    total_successful = 0
    total = 0

    cv.namedWindow('Zpiceny eduroam', 0)
    for img_name in test_images:
        s, t = process_img(
                img_name=img_name,
                signaller=lambda pic: cnn(pic)[0],
                classifier=None,
                pkm_coordinates=pkm_coordinates)
        total_successful += s
        total += t

    print(f'Total success rate: {total_successful / total}')


@click.command()
@click.option('--lbp-model', help='Path to XGB model used in prediction')
@click.option('--hog-model', help='Path to SVM model used in prediction')
@click.option('--edge-model', help='Path to model used in prediction ' +
              'derived from detected edges')
@click.option('--classifier-model', help='Path to MLP model')
def classify(lbp_model,
             hog_model,
             edge_model,
             classifier_model) -> None:

    pkm_coordinates = util.load_coords('data/parking_map_python.txt')
    test_images = sorted([img for img in glob.glob('data/test_images/*.jpg')])
    lbp_booster = util.load_booster(lbp_model)
    hog_svm = cv.ml.SVM.load(hog_model)
    edge_pred = EdgePredictor.from_file(edge_model)

    signaller = CombinedSignaller(lbp=lbp_booster, hog=hog_svm,
                                  edge_pred=edge_pred)
    classifier = util.load_mlp(classifier_model)

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


@click.command()
@click.option('--highlight-cars', type=bool, default=False)
def rcnn(highlight_cars) -> None:
    rcnn_detection(highlight_cars)


@click.group('Classification')
def main() -> None:
    pass


main.add_command(classify)
main.add_command(cnn_classify)
main.add_command(rcnn)
main.add_command(resnet_classify)


if __name__ == "__main__":
    main()
