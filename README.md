
## Train

```sh
python3 train.py train-lbp --free-set train_images/free --occupied-set train_images/full --model-name lbp.json
python3 train.py train-hog --free-set train_images/free --occupied-set train_images/full --model-name hog.yml

python3 parking_lot/train.py train-final-classifier --free-set data/train_images/free --occupied-set data/train_images/full --hog-model models/hog.yml --lbp-model models/lbp.json --model-name models/classifier.model
```

## Predict

```sh
python3 parking_lot classify --lbp-model models/lbp.json --hog-model models/hog.yml --final-classifier-model models/classifier.model
```

