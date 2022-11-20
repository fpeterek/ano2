import math

import cv2 as cv
import numpy as np

import util


class EdgeSignaller:

    def __init__(self):
        self.non_zero = 0
        self.longest_line = 0
        self.img = None

        self.eps = 3
        self.min_pts = 3

    @property
    def img_height(self) -> int:
        return self.img.shape[0]

    @property
    def img_width(self) -> int:
        return self.img.shape[1]

    @property
    def img_pixels(self) -> int:
        return self.img_height * self.img_width

    # Simplify bounds checking
    def white_px(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.img_width or y >= self.img_height:
            return False
        return self.img[y, x] > 127

    def reset(self):
        self.non_zero = 0
        self.longest_line = 0
        self.img = None

    def count_non_zero(self) -> float:
        for y in range(self.img_height):
            for x in range(self.img_width):
                self.non_zero += self.white_px(x, y)

    def non_zero_signal(self):
        non_zero_ratio = self.non_zero / self.img_pixels

        return min(non_zero_ratio, 0.1) / 0.1

    def color_line(self, x, y, labels, color) -> None:
        if x < 0 or y < 0 or x >= self.img_width or y >= self.img_height:
            return
        if labels[y][x]:
            return
        if not self.white_px(x, y):
            return

        labels[y][x] = color

        for i in range(8):
            angle = math.pi * (i / 4)
            dx = round(math.cos(angle))
            dy = round(math.sin(angle))
            self.color_line(x+dx, y+dy, labels, color)

    def find_longest_line(self):
        used = {0}
        labels = [[0 for _ in range(self.img_width)] for _
                  in range(self.img_height)]

        for x in range(self.img_width):
            for y in range(self.img_height):
                if labels[y][x]:
                    continue
                if self.white_px(x, y):
                    next_color = max(used) + 1
                    used.add(next_color)
                    self.color_line(x, y, labels, next_color)

        lengths = {0: 0}

        for x in range(self.img_width):
            for y in range(self.img_height):
                color = labels[y][x]
                if color:
                    lengths[color] = lengths.get(color, 0) + 1

        self.longest_line = max(lengths.values())
        # print(f'longest_line={self.longest_line}')

    def longest_line_signal(self):
        max_len = (self.img_pixels * 0.015)
        return min(self.longest_line, max_len) / max_len

    def is_core(self, x, y):
        whites = 0
        for cx in range(x-self.eps, x+self.eps+1):
            for cy in range(y-self.eps, y+self.eps+1):
                whites += (cx, cy) != (x, y) and self.white_px(cx, cy)

        return whites >= self.min_pts

    def scan_px(self, x, y, labels, color, core_check=True) -> bool:

        if not self.white_px(x, y) or labels[y][x]:
            return False

        if core_check and not self.is_core(x, y):
            return False

        labels[y][x] = color

        lx = max(x-self.eps, 0)
        rx = min(x+self.eps+1, self.img_width)
        uy = max(y-self.eps, 0)
        ly = min(y+self.eps+1, self.img_height)

        for cx in range(lx, rx):
            for cy in range(uy, ly):
                if (cx, cy) == (x, y) or not self.white_px(cx, cy):
                    continue
                if self.is_core(cx, cy):
                    self.scan_px(cx, cy, labels, color, core_check=False)
                elif not labels[cy][cx]:
                    labels[cy][cx] = color

        return True

    def m(self, labels, p, q, color) -> float:
        total = 0
        for x in range(self.img_width):
            xp = x**p
            for y in range(self.img_height):
                total += xp * y**q * (labels[y][x] == color)
        return total

    def mu(self, labels, xt, yt, p, q, color) -> float:
        total = 0
        for x in range(self.img_width):
            xp = (x - xt)**p
            for y in range(self.img_height):
                yp = (y-yt)**q
                total += xp * yp * (labels[y][x] == color)
        return total

    def compute_bounding_box_signal(self, labels, color) -> float:
        l_x = len(labels[0])
        r_x = 0
        u_y = len(labels)
        l_y = 0
        for y, row in enumerate(labels):
            for x, label in enumerate(row):
                if label == color:
                    l_x = min(l_x, x)
                    r_x = max(r_x, x)
                    u_y = min(u_y, y)
                    l_y = max(l_y, y)

        width = r_x - l_x
        height = l_y - u_y

        if width <= 0 or height <= 0:
            return 0.0

        area = width * height
        img_area = len(labels[0]) * len(labels)

        return min(0.3, area / img_area) / 0.3

    def compute_cluster_signals(self, labels, color) -> list[float]:
        m00 = self.m(labels, 0, 0, color)
        m10 = self.m(labels, 1, 0, color)
        m01 = self.m(labels, 0, 1, color)

        xt = m10 / m00
        yt = m01 / m00

        mu20 = self.mu(labels, xt, yt, 2, 0, color)
        mu02 = self.mu(labels, xt, yt, 0, 2, color)
        mu11 = self.mu(labels, xt, yt, 1, 1, color)

        left = 0.5 * (mu20 + mu02)
        right = 0.5 * math.sqrt(4 * mu11**2 + (mu20 - mu02)**2)

        mu_max = left + right
        mu_min = left - right

        sig1 = mu_min / mu_max if mu_max != 0 else 0

        return [sig1, self.compute_bounding_box_signal(labels, color)]

    def cluster_signals(self):
        current_color = 1
        labels = [[0 for _ in range(self.img_width)] for _
                  in range(self.img_height)]
        for x in range(self.img_width):
            for y in range(self.img_height):
                current_color += self.scan_px(x, y, labels, current_color)

        counts = {0: 0}

        for x in range(self.img_width):
            for y in range(self.img_height):
                color = labels[y][x]
                if color:
                    counts[color] = counts.get(color, 0) + 1

        largest_cluster = 0

        for k, v in counts.items():
            if v > counts[largest_cluster]:
                largest_cluster = k

        if not largest_cluster:
            return [0.0, 0.0]

        # print('[')
        # for row in labels:
        #     row = ''.join([str(i) if i else ' ' for i in row])
        #     print(f'    {row}')
        # print(']')

        return self.compute_cluster_signals(labels, largest_cluster)

    def predict(self, img) -> list[float]:
        self.reset()
        self.img = cv.Canny(img, 150, 200)

        self.count_non_zero()
        # self.find_longest_line()

        # sigs = [self.non_zero_signal(),
        #         self.longest_line_signal(),
        #         ] + self.cluster_signals()

        # return sigs
        return [self.non_zero_signal()]

    def __call__(self, img) -> list[float]:
        return self.predict(img)


class EdgePredictor:
    def __init__(self, mlp):
        self.signaller = EdgeSignaller()
        self.mlp = mlp

    def predict(self, img):
        sigs = self.signaller(img)
        return sigs[0]
        # mlp_pred = self.mlp.predict(np.asarray([sigs]))
        # return mlp_pred[0]

    @staticmethod
    def from_file(filename: str):
        # mlp = util.load_mlp(filename)
        return EdgePredictor(None)
