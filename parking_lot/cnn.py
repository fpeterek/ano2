import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.models as models

import util


# TODO: Dropout

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.norm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = F.relu(self.norm3(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def from_file(filename):
        net = Net()
        net.load_state_dict(torch.load(filename))
        return net


class CNNSignaller:
    def __init__(self, model):
        self.model = None

        if isinstance(model, str):
            self.model = Net.from_file(model)
        elif isinstance(model, Net):
            self.model = model
        else:
            raise TypeError('Invalid value for argument model',
                            f'({type(model)})')

        self.model.eval()
        self.transform = util.cnn_transform()

    def predict(self, img):
        img = Image.fromarray(np.uint8(img))
        transformed = self.transform(img).unsqueeze(0)
        prob, label = F.softmax(self.model(transformed), dim=1).topk(1)
        prob = float(prob[0][0])
        label = int(label[0][0])
        prob = prob if label else 1-prob

        return label  # prob


class ResnetSignaller:
    def __init__(self, model):
        self.model = None

        if isinstance(model, str):
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2)
            self.model.load_state_dict(torch.load(model))
        elif isinstance(model, models.ResNet):
            self.model = model
        else:
            raise TypeError('Invalid value for argument model')

        self.model.eval()
        self.transform = util.resnet_transform()

    def predict(self, img):
        img = Image.fromarray(np.uint8(img))
        transformed = self.transform(img).unsqueeze(0)
        prob, label = F.softmax(self.model(transformed), dim=1).topk(1)
        # print(prob)
        # print(label)

        prob = float(prob[0][0])
        label = int(label[0][0])
        prob = prob if label else 1-prob

        return label  # prob
