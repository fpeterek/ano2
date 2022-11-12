import xgboost as xgb
import numpy as np

import util


class CombinedSignaller:

    def lbp_predict(self, img):
        hist = xgb.DMatrix(np.matrix(self.lbp(img)))
        return self.lbp_model.predict(hist)[0]

    def hog_predict(self, img):
        hog_sigs = self.hog.compute(img)
        return self.hog_model.predict(np.matrix(hog_sigs))[1][0][0]

    def cnn_predict(self, img):
        return self.cnn.predict(img)

    def __init__(self, hog, lbp, cnn, edge_pred):
        self.hog = util.create_hog_descriptor()
        self.lbp = util.create_lbp_signaller()
        self.hog_model = hog
        self.lbp_model = lbp
        self.cnn = cnn
        self.edge_pred = edge_pred

    def get_signals(self, img) -> np.array:
        cnn = self.cnn_predict(img)
        lbp = self.lbp_predict(img)
        hog = self.hog_predict(img)
        edge = self.edge_pred.predict(img)

        sigs = [lbp, hog, edge]

        return np.array(sigs)
