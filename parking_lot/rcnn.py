import glob

import torchvision.models.detection as detection
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv

import util


def rcnn(highlight_cars) -> None:
    pkm_coordinates = util.load_coords('data/parking_map_python.txt')
    test_images = sorted([img for img in glob.glob('data/test_images/*.jpg')])

    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()])

    classes = ['__background__', 'person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
               'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife',
               'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
               'chair', 'couch', 'potted plant', 'bed', 'N/A',
               'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
               'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']

    cv.namedWindow('rccn window', 0)
    for img in test_images:
        img = cv.imread(img)
        cpy = img.copy()
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        img_rcnn = transform(img_pil).unsqueeze(0)
        outputs = model(img_rcnn)

        pred = [classes[i] for i in outputs[0]['labels'].numpy()]
        bounds = outputs[0]['boxes'].detach().numpy()

        cars = []

        for c, boundary in zip(pred, bounds):
            if c not in ('car', 'motorcycle', 'truck', 'bus'):
                continue
            if highlight_cars:
                highlight_car(cpy, boundary)
            cars.append(boundary)

        res = []

        for space in pkm_coordinates:
            occ = is_occupied(space, cars, 0.3)
            res.append(occ)

            if occ:
                util.mark_occupied(space, cpy)

        cv.imshow('rccn window', cpy)
        cv.waitKey(0)


def is_occupied(space: list, cars: list, threshold: int):
    space = [
        min(space[0], space[2], space[4], space[6]),
        min(space[1], space[3], space[5], space[7]),
        max(space[0], space[2], space[4], space[6]),
        max(space[1], space[3], space[5], space[7]),
        ]

    for car in cars:
        occupied = iou(space, car) > threshold
        if occupied:
            return True

    return False


def iou(space: list, car: list):
    intersect = intersection(space, car)
    union = area(space) + area(car) - intersect

    return intersect / union


def area(rect: list):
    x = rect[2] - rect[0]
    y = rect[3] - rect[1]

    return x*y


def intersection(space: list, car: list):
    left = space if space[0] < car[0] else car
    right = space if left is car else car

    if left[2] < right[0]:
        return 0

    left_x = right[0]
    right_x = min(right[2], left[2])

    int_x = max(right_x - left_x, 0)

    top = space if space[1] < car[1] else car
    bottom = space if top is car else car

    if top[3] < bottom[1]:
        return 0

    top_y = bottom[1]
    bottom_y = min(bottom[3], top[3])

    int_y = max(bottom_y - top_y, 0)

    return int_x * int_y


def highlight_car(img, bounds):
    start = int(bounds[0]), int(bounds[1])
    end = int(bounds[2]), int(bounds[3])
    cv.rectangle(img, start, end, (0, 0, 255), 3)
