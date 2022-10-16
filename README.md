
## Train

```sh
python3 train.py train-lbp --free-set train_images/free --occupied-set train_images/full --model-name lbp.json
python3 train.py train-hog --free-set train_images/free --occupied-set train_images/full --model-name hog.yml
```

## Predict

```sh
python3 main.py lbp-classifier --xgb-model lbp.json
python3 main.py hog-classifier --model-name hog.yml --enable-edge-detection y
```

