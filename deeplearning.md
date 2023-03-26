# Detection Of Occupied Parking Slots

This program focuses on detection of occupied parking slots. Multiple approaches of image analysis
are compared. 

## Dataset

[Dataset source](http://mrl.cs.vsb.cz//people/fusek/ano2_course.html)

The training dataset consists of pictures of empty and occupied parking slots. There are `2968`
pictures of empty parking slots and `1239` pictures of occupied slots, thus we can say the dataset
is somewhat imbalanced.

The testing dataset consists of pictures of an entire parking lot, along with coordinates
specifying a (possibly skewed) bounding box of each slot, as well as a file marking occupied slots.
The testing dataset consists of pictures of (nearly) fully occupied lots, semi-occupied lots, and 
empty or nearly empty lots. The testing dataset is reasonably balanced.

## Evaluation Metric

We will be using the accuracy metric to evaluate our models. The reasons being:

* The imbalance isn't that strong
* The accuracy metric is simple to use and understand
* Accuracy appears like a reasonable metric for this concrete usecase

Yes, if we predicted each parking slot as unoccupied, and there was only one occupied parking slot
in the lot, we would deem our model as accurate, even if it was absolutely terrible. However, we
also have multiple testing images of (nearly) fully occupied lots. The testing dataset is somewhat
balanced, so we will get a reasonable insight into the performance of our model by using the
accuracy metric.

## Preprocessing

For all approaches except RCNN, input images are converted to grayscale. Then, parking slots are
extracted from the images and transformed so as to remove any skewing introduced by the angle of
the camera relative to the parking space.

## Traditional Approach

First, we try to approach the task using a set of traditional image processing methods. For each
slot, we compute a histogram of oriented gradients, an LBP histogram, and the ratio of edge
pixels to one tenth of non-edge pixels (Canny detection is applied to obtain the edge pixels).

We then use a Support Vector Machine to predict whether a parking space is occupied from the HOG
signals. An XGBoost forest classifier is used to make a prediction from the LBP histogram. 
The ratio of edge pixels found in the input image is clipped to fit in the range `[0; 1]`.
Afterwards, the probabilities predicted by the two ML classifiers, along with the clipped ratio,
are used as the input of a Multi-Layer Perceptron to perform the final prediction and determine
whether the parking space is occupied.

This approach is somewhat slow at around two seconds per parking lot on my device (i5-12600KF),
though it does perform with a reasonable degree of accuracy.

The following output describes the accuracies for each testing image, along with a total accuracy
across all testing images.

```
Success rate: 1.0
Success rate: 1.0
Success rate: 0.9821428571428571
Success rate: 0.9464285714285714
Success rate: 1.0
Success rate: 0.9642857142857143
Success rate: 0.9821428571428571
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 0.9642857142857143
Success rate: 0.9642857142857143
Success rate: 0.9821428571428571
Success rate: 0.9821428571428571
Success rate: 0.9821428571428571
Success rate: 1.0
Success rate: 0.9107142857142857
Success rate: 0.9464285714285714
Success rate: 0.9642857142857143
Success rate: 0.9642857142857143
Success rate: 0.9821428571428571
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Total success rate: 0.9799107142857143
```

Relevant files:

[parking_lot/__main__.py](parking_lot/__main__.py)
[parking_lot/combined_signaller.py](parking_lot/combined_signaller.py)
[parking_lot/util.py](parking_lot/util.py)
[parking_lot/edge_predictor.py](parking_lot/edge_predictor.py)
[parking_lot/train.py](parking_lot/train.py)

## CNN

Next, we explore the option of classifying the input images using a convolutional neural network.

Our neural network consists of three convolutional layers, with all three being succeeded by
a batch normalization layer. Then, the ReLU activation function is applied. We also apply max
pooling between the first and second, and second and third convolutional layers. There is no
pooling after the last convolutional layer. The fully connected neural network consists of three
layers. Dropout is utilized in the fully connected portion of the CNN.

We always store the model which performed the best.

Initially, I attempted to present the network not with the original input images, but with images
representing local binary patterns located in the input images. For smaller CNNs, this increased
accuracy by up to 15-20 percentage points, mainly in darker input images. However, as the CNN grew
in size and the architecture got more complicated, the LBP images eventually started to have
a negative effect.

The trained network, however, still had problems with very bright and very dark images, as well as
images which contained a lot of noise (e.g. shadows). This could, however, be rectified using data
enhancement techniques.

### Data Enhancement

Using data enhancement, we increase the size of the training dataset sixfold.

The first transformation is the identity transformation -- we keep the original image as it was.

For our second and third transformation, we darken the image, in each case to a different degree.
The forth transformation then lightens the image. All three images are also rotated by a random
degree.

The fifth transformation adds shadows to the input image. There are multiple ways shadows are
applied to the input. The first way adds a rectangular shade to a randomly selected side of the
image. The second way generates the shade using the `sin` function to obtain a wave, rather than
a box. Frequency and direction are randomized. The last technique of adding shadows is to apply
a mask. The dataset contains pictures where shadows cast by trees cause problems and confuse the
neural network. Thus, we select a sufficiently complicated mask (in our case it is a leaf which
was freely available in my image editor of choice), and darken the input image wherever the mask
is set to zero. The technique of shadow application is also selected randomly.

The last transformation deals with lamps and lampposts located in parking lots. The input images
contained lampposts, which could occasionally confuse the network. To counteract the effect
of lampposts, we add lampposts to our training data, to teach the network to ignore lamps.
We randomly add either an ellipse representing the top of the lamp post, or a long but narrow
rectangle representing the post, to our input images. The sixth image is generated by
unconditionally adding a lamppost to the input image and then rotating the image by a random
degree. We also randomly add lampposts to some of the images generated during the second to fourth
step of the data enhancement process.

Using the described architecture and data enhancement process, we are able to get a very accurate
model.

```
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 0.9464285714285714
Success rate: 1.0
Success rate: 1.0
Success rate: 0.9642857142857143
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 1.0
Success rate: 0.9464285714285714
Success rate: 1.0
Success rate: 0.9642857142857143
Success rate: 1.0
Success rate: 1.0
Success rate: 0.9821428571428571
Success rate: 0.9821428571428571
Success rate: 1.0
Total success rate: 0.9910714285714286
```

Relevant files:

[parking_lot/__main__.py](parking_lot/__main__.py)
[parking_lot/util.py](parking_lot/util.py)
[parking_lot/train.py](parking_lot/train.py)
[parking_lot/cnn.py](parking_lot/cnn.py)
[parking_lot/data_enhancement.py](parking_lot/data_enhancement.py)

## Transfer Learning

We also attempt to apply transfer learning techniques. We load a pretrained `ResNet18` network
and modify the very last layer in the fully connected portion of the network. We then train the
network. No fine-tuning and training of the entire network is applied.

Whilst transfer learning proved to be a quick and simple way to get a working model, the resulting
network is not as accurate as the previous two classifiers, with a total success rate of slightly
below 96 %.

It is important to note that we didn't train the ResNet classifier on the enhanced dataset.
However, the classifier which used traditional methods of image analysis was also trained on the
original unenhanced dataset, and performed better.


```
Success rate: 1.0
Success rate: 1.0
Success rate: 0.9821428571428571
Success rate: 0.9285714285714286
Success rate: 0.9642857142857143
Success rate: 1.0
Success rate: 0.9285714285714286
Success rate: 1.0
Success rate: 0.9285714285714286
Success rate: 0.9464285714285714
Success rate: 0.9821428571428571
Success rate: 0.9642857142857143
Success rate: 0.9642857142857143
Success rate: 0.8928571428571429
Success rate: 0.9107142857142857
Success rate: 0.9107142857142857
Success rate: 1.0
Success rate: 0.8571428571428571
Success rate: 0.9642857142857143
Success rate: 0.9285714285714286
Success rate: 0.9642857142857143
Success rate: 1.0
Success rate: 0.9821428571428571
Success rate: 0.9821428571428571
Total success rate: 0.9575892857142857
```

Relevant files:

[parking_lot/__main__.py](parking_lot/__main__.py)
[parking_lot/util.py](parking_lot/util.py)
[parking_lot/train.py](parking_lot/train.py)
[parking_lot/data_enhancement.py](parking_lot/data_enhancement.py)

## Region Based CNN

For our last experiment, we try to utilize a pretrained Region Based CNN. We use the
`fasterrcnn_resnet50_fpn` network.

First, the network is used to detect objects in the input image. Then, detections of unexpected
objects are filtered out. It is very unlikely that somebody would park their carrot at a university
parking lot. And even if they did, the carrot would probably not be large enough so that a car
couldn't park over the carrot without damaging it.

Afterwards, we compute the intersection over union ratio for each detected object and parking
space. If the ratio is greater than a specified threshold, the slot is marked as occupied.

However, since the training set of the utilized network most likely contained images of cars shot
with the camera located on the same level as the car, not from a birds-eye view, and since the
network is mostly a general purpose network, the accuracy was obviously subpar on the first glance
and we didn't explore this option further.

Relevant files:

[parking_lot/__main__.py](parking_lot/__main__.py)
[parking_lot/rcnn.py](parking_lot/rcnn.py)
