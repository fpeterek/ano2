# Parking Lot Occupation Prediction

All relevant source code is located in the `parking_lot` folder.

All pretrained models are located in the `models` folder.


# Data Enhancement

```sh
# Create the necessary folders
mkdir data/enhanced
mkdir data/enhanced/{free,full}

# Run the enhancement algorithm
# Set vis to y if you want to visualize the enhanced dataset
python3 parking_lot/data_enhancement.py enhance --free data/train_images/free --occupied data/train_images/full --dest data/enhanced --vis n
```

# Traditional Methods

Classification using traditional methods combines HOG, LBP histograms and edge detection to detect whether a parking
space is occupied.

An XGBoost model is used to predict whether a parking space is occupied using LBP.

A Support Vector Machine is used in conjunction with a Histogram of Oriented Gradients.

The Canny edge detector is used to detect edges. The probability of a space being occupied is the total number
of edge pixels divided by one tenth of the size of the entire image, capped at 1.0.

In the end, those three signals are combined using a neural network, which has a final word in predicting whether
a parking space is occupied or not.

## Training

```sh
# Train the LBP model
# The file should have .json suffix as the model should be saved in JSON format
python3 parking_lot/train.py train-lbp --free-set train_images/free --occupied-set train_images/full --model-name lbp.json

# Train the HOG model
python3 parking_lot/train.py train-hog --free-set train_images/free --occupied-set train_images/full --model-name hog.yml

# Train the final classifier, which takes the result of the HOG, LBP and edge prediction and gives us
# a rather accurate result
# The suffix does not matter as the neural network is pickled
python3 parking_lot/train.py train-final-classifier --free-set data/train_images/free --occupied-set data/train_images/full --hog-model models/hog.yml --lbp-model models/lbp.json --model-name models/classifier.model
```

## Prediction

```sh
python3 parking_lot classify --lbp-model models/lbp.json --hog-model models/hog.yml --final-classifier-model models/classifier.model
```

# Custom Convolutional Neural Network

A custom CNN is used to predict whether a parking space is occupied or not. The CNN should be trained on the enhanced dataset
to obtain reasonable results.

## Training

```sh
# File suffix does not matter as the model is pickled
python3 parking_lot/train.py train-cnn --free-set data/enhanced/free --occupied-set data/enhanced/full --model-name 'cnn.model'
```

## Prediction

```sh
python3 parking_lot cnn-classify --cnn-model models/cnn.model
```

# Resnet

## Training

```sh
python3 parking_lot/train.py train-resnet --free-set data/enhanced/free --occupied-set data/enhanced/full --model-name 'resnet.pt'
```

## Prediction

```sh
python3 parking_lot resnet-classify --resnet-model models/resnet.pt
```

# Region-Based Convolutional Neural Network

```sh
python3 parking_lot rcnn --highlight-cars y
```

