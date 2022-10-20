import pickle

import click
import cv2 as cv
import numpy as np
import xgboost as xgb
import sklearn.neural_network as sknn

import util
from combined_signaller import CombinedSignaller


@click.command()
@click.option('--free-set', help='Folder with images of free parking spaces')
@click.option('--occupied-set',
              help='Folder with images of occupied parking spaces')
@click.option('--model-name', help='Name of model')
@click.option('--c', default=100, help='C parameter of svm')
@click.option('--gamma', default=1.0, help='Gamma parameter of svm')
def train_hog(free_set: str, occupied_set: str, model_name: str, c: float,
              gamma: float):
    hog = util.create_hog_descriptor()

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_INTER)
    svm.setC(c)
    svm.setGamma(gamma)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    signals, labels = util.load_training_ds(
            signaller=hog.compute,
            free_folder=free_set,
            occupied_folder=occupied_set)

    signals = np.matrix(signals)
    labels = np.array(labels)

    svm.train(signals, cv.ml.ROW_SAMPLE, labels)

    svm.save(model_name)


@click.command()
@click.option('--free-set', help='Folder with images of free parking spaces')
@click.option('--occupied-set',
              help='Folder with images of occupied parking spaces')
@click.option('--model-name', help='Name of model')
def train_lbp(free_set: str, occupied_set: str, model_name: str):
    signaller = util.create_lbp_signaller()

    signals, labels = util.load_training_ds(
            signaller=signaller,
            free_folder=free_set,
            occupied_folder=occupied_set)

    signals = np.matrix(signals)
    labels = np.array(labels)

    dtrain = xgb.DMatrix(data=signals, label=labels)

    params = {
            # 'max_depth': 4,
            # 'eta': 0.8,
            'max_depth': 5,
            'eta': 0.8,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
    }

    rounds = 10

    booster = xgb.train(params, dtrain, rounds)

    booster.save_model(model_name)


@click.command()
@click.option('--free-set', help='Folder with images of free parking spaces')
@click.option('--occupied-set',
              help='Folder with images of occupied parking spaces')
@click.option('--hog-model', help='Name of HOG model')
@click.option('--lbp-model', help='Name of LBP model')
@click.option('--model-name', help='Name of model')
def train_final_classifier(
        free_set: str,
        occupied_set: str,
        hog_model: str,
        lbp_model: str,
        model_name: str):

    lbp_model = util.load_booster(lbp_model)
    hog_model = cv.ml.SVM.load(hog_model)
    signaller = CombinedSignaller(hog=hog_model, lbp=lbp_model)

    signals, labels = util.load_training_ds(
            signaller=signaller.get_signals,
            free_folder=free_set,
            occupied_folder=occupied_set)

    signals = np.asarray(signals)
    labels = np.array(labels)

    mlp = sknn.MLPClassifier(
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(4, 4),
            random_state=350)

    mlp.fit(signals, labels)

    with open(model_name, 'wb') as file:
        pickle.dump(mlp, file)


@click.group('Training')
def main() -> None:
    pass


main.add_command(train_hog)
main.add_command(train_lbp)
main.add_command(train_final_classifier)


if __name__ == '__main__':
    main()
