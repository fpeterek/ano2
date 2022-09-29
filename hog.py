import glob

import cv2 as cv
import numpy as np

train_images_full = []


train_images_list = []
train_labels_list = []

img_size = 96
hog = cv.HOGDescriptor((img_size, img_size), (32, 32), (16, 16), (8, 8), 9, 1,
                       -1, 0, 0.2, 1, 64, True)

svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_INTER)
svm.setC(100.0)
svm.setTermCriteria(cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)


for img_name in train_images_full:
    one_slot_img = cv.imread(img_name, 0)
    res_img = cv.resize(one_slot_img, (img_size, img_size))
    hog_feature = hog.compute(res_img)
    train_images_list.append(hog_feature)
    train_images_list.append(1)


for img_name in train_images_full:
    one_slot_img = cv.imread(img_name, 0)
    res_img = cv.resize(one_slot_img, (img_size, img_size))
    hog_feature = hog.compute(res_img)
    train_images_list.append(hog_feature)
    train_images_list.append(0)

svm.train(np.array(train_images_list), cv.ml.ROW_SAMPLE,
          np.array(train_labels_list))

svm.save('my.xml')

test_img = [img for img in glob.glob('test_images/*.jpg')]
test_img.sort()

result_list = []

for img in test_img:
    one_slot_img = cv.imread(img)
    img_copy = one_slot_img.copy()
