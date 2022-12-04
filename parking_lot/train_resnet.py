import torch

import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

from torch_ds import CarParkDS
import util as util


resnet18 = models.resnet18(pretrained=True)

num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)

transform = util.resnet_transform()

batch_size = 4

trainset = CarParkDS(occupied_dir='data/enhanced/full',
                     empty_dir='data/enhanced/free',
                     transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

for epoch in range(8):  # loop over the dataset multiple times

    print(f'{epoch=}')

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(resnet18.state_dict(), 'models/resnet.pt')
