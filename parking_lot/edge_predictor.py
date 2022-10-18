import cv2 as cv


class EdgePredictor:

    def predict(self, img) -> float:
        canny_img = cv.Canny(img, 150, 200)

        non_zero_ratio = 0

        for y in range(canny_img.shape[0]):
            for x in range(canny_img.shape[1]):
                non_zero_ratio += canny_img[y, x] > 127

        non_zero_ratio /= img.shape[0] * img.shape[1]

        non_zero_ratio = min(non_zero_ratio, 0.1)

        return non_zero_ratio / 0.1
