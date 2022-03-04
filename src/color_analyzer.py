from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import io
import time


def convert_to_polar(value):
    hue, sat = value
    hue = hue * 2 * math.pi
    return (hue, sat)


def convert_to_rgb(value):
    value = value
    rgb = np.round(np.multiply(colorsys.hsv_to_rgb(*value), 255))
    return tuple(int(x) for x in rgb)


def rgb_to_hex(rgb):
    return "%02x%02x%02x" % rgb


def rgb_to_css(rgb):
    return f"rgb({','.join([str(x) for x in rgb])})"


def open_image(bytes, max_size):
    image = Image.open(io.BytesIO(bytes))
    image.thumbnail(max_size)

    return np.array(image.convert("HSV").getdata())


class ColorAnalyzer:
    def __init__(self, image):
        self.max_size = (256, 256)
        self.epsilon = 3.0
        self.min_samples = 6

        self.image = open_image(image, self.max_size)

    def calculate_proportions(self):
        img = self.image

        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_samples).fit(img)
        dbscan.fit(img)

        labels = dbscan.labels_
        clusters = {}
        for pixel, cluster in zip(img, labels):
            if cluster != -1:
                if cluster not in clusters:
                    clusters[cluster] = []

                clusters[cluster].append(pixel)

        result = {}
        for cluster in clusters:
            hsv = np.average(clusters[cluster], axis=0) / 255
            rgb = convert_to_rgb(hsv)
            css = rgb_to_css(rgb)
            hexcolor = rgb_to_hex(rgb)
            polar = convert_to_polar(hsv[:-1])
            count = len(clusters[cluster])

            result[f"{cluster}"] = {
                "rgb": rgb,
                "css": css,
                "hex": hexcolor,
                "polar": polar,
                "count": min(count, 50),
            }

        return result

    def get_dominant_colors(self, clusters):
        n = 5
        dominant = [
            x["rgb"] for x in sorted(clusters.values(), key=lambda x: x["count"])[:n]
        ]
        print(dominant)
        return [{"rgb": rgb_to_css(x), "hex": rgb_to_hex(x)} for x in dominant]
