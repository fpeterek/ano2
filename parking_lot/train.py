import pickle

import click
import cv2 as cv
import numpy as np
import xgboost as xgb
import sklearn.neural_network as sknn
import torch
import torch.optim as optim
import torch.nn as nn

from cnn import Net

from torch_ds import CarParkDS
import util
from combined_signaller import CombinedSignaller
from edge_predictor import EdgeSignaller, EdgePredictor
from cnn import CNNSignaller


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
@click.option('--free-set', help='Free parking spaces')
@click.option('--occupied-set', help='Occupied parking spaces')
@click.option('--epochs', help='Number of training epochs')
@click.option('--model-name', help='Output path')
def train_cnn(
        free_set: str,
        occupied_set: str,
        epochs: int,
        model_name: str,
        ):

    transform = util.cnn_transform()

    batch_size = 4

    trainset = CarParkDS(occupied_dir=occupied_set,
                         empty_dir=free_set,
                         transform=transform)
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    best_loss = 1.0
    best_model = {}

    for epoch in range(3):  # loop over the dataset multiple times

        print(f'{epoch=}')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}]',
                      f'loss: {running_loss / 200:.3f}')
                if running_loss < best_loss:
                    best_model = net.state_dict().copy()
                running_loss = 0.0

    print('Finished Training')

    torch.save(best_model, model_name)


@click.command()
@click.option('--free-set', help='Folder with images of free parking spaces')
@click.option('--occupied-set',
              help='Folder with images of occupied parking spaces')
@click.option('--model-name', help='Name of model')
def train_edge_classifier(
        free_set: str,
        occupied_set: str,
        model_name: str):

    signaller = EdgeSignaller()

    signals, labels = util.load_training_ds(
            signaller=signaller,
            free_folder=free_set,
            occupied_folder=occupied_set)

    signals = [np.array(sigs) for sigs in signals]
    signals = np.asarray(signals)
    labels = np.array(labels)

    mlp = sknn.MLPClassifier(
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(6, ),
            random_state=350)

    mlp.fit(signals, labels)

    with open(model_name, 'wb') as file:
        pickle.dump(mlp, file)


@click.command()
@click.option('--free-set', help='Folder with images of free parking spaces')
@click.option('--occupied-set',
              help='Folder with images of occupied parking spaces')
@click.option('--cnn-model', help='Name of CNN model')
@click.option('--hog-model', help='Name of HOG model')
@click.option('--lbp-model', help='Name of LBP model')
@click.option('--edge-model', help='Name of edge model')
@click.option('--model-name', help='Name of model')
def train_final_classifier(
        free_set: str,
        occupied_set: str,
        cnn_model: str,
        hog_model: str,
        lbp_model: str,
        edge_model: str,
        model_name: str):

    cnn = CNNSignaller(cnn_model)
    lbp_model = util.load_booster(lbp_model)
    hog_model = cv.ml.SVM.load(hog_model)
    edge_pred = EdgePredictor.from_file(edge_model)
    signaller = CombinedSignaller(cnn=cnn, hog=hog_model, lbp=lbp_model,
                                  edge_pred=edge_pred)

    signals, labels = util.load_training_ds(
            signaller=signaller.get_signals,
            free_folder=free_set,
            occupied_folder=occupied_set)

    signals = np.asarray(signals)
    labels = np.array(labels)

    mlp = sknn.MLPClassifier(
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(8, ),
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
main.add_command(train_edge_classifier)
main.add_command(train_cnn)


if __name__ == '__main__':
    main()
