import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import cv2 as cv

from nn import Net


transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ds_size = len(testset)
print(f'{ds_size=}')

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# cv.namedWindow('Jebat Go', 0)
# for i, img in enumerate(images):
#     print(f'Label: {classes[labels[i]]}')
#     cv.imshow('Jebat Go', img.numpy().transpose(1, 2, 0))
#     cv.waitKey(0)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# print('Training...')

if False:
    for epoch in range(4):  # loop over the dataset multiple times

        print(f'{epoch=}')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), 'models/cnn.pt')

net.load_state_dict(torch.load('models/cnn.pt'))

dataiter = iter(testloader)
images, labels = dataiter.next()

outputs = net(images)

_, predictions = torch.max(outputs, 1)

cv.namedWindow('Jebat Go', 0)
for pred, img in zip(predictions, images):
    print(f'Label: {classes[pred]}')
    cv.imshow('Jebat Go', img.numpy().transpose(1, 2, 0))
    cv.waitKey(0)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images:',
      f'{100 * correct // total} %')
