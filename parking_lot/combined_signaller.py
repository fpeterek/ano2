import xgboost as xgb
import numpy as np

import util
from edge_predictor import EdgePredictor


class CombinedSignaller:

    def lbp_predict(self, img):
        hist = xgb.DMatrix(np.matrix(self.lbp(img)))
        return self.lbp_model.predict(hist)[0]

    def hog_predict(self, img):
        hog_sigs = self.hog.compute(img)
        return self.hog_model.predict(np.matrix(hog_sigs))[1][0][0]

    def __init__(self, hog, lbp):
        self.hog = util.create_hog_descriptor()
        self.lbp = util.create_lbp_signaller()
        self.hog_model = hog
        self.lbp_model = lbp
        self.edge_pred = EdgePredictor()

    def get_signals(self, img) -> np.array:
        lbp = self.lbp_predict(img)
        hog = self.hog_predict(img)
        edge = self.edge_pred.predict(img)

        sigs = [lbp, hog] + edge

        return np.array(sigs)
