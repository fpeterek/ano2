
## Train

```sh
python3 train.py train-lbp --free-set train_images/free --occupied-set train_images/full --model-name lbp.json
python3 train.py train-hog --free-set train_images/free --occupied-set train_images/full --model-name hog.yml
```

## Predict

```sh
python3 parking_lot classify --lbp-model models/lbp.json --hog-model models/hog.yml
```

