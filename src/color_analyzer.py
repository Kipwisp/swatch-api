from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
import colorsys
import math
import io


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


def get_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def open_image(bytes, max_size):
    image = Image.open(io.BytesIO(bytes))
    image.thumbnail(max_size)

    return np.array(image.convert("HSV").getdata())


class ColorAnalyzer:
    def __init__(self, image):
        self.max_size = (128, 128)
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

        max = sum([len(clusters[cluster]) for cluster in clusters])

        n = 7
        dist_threshold = 70
        cov_threshold = 0.005
        palette = []
        result = {}
        for cluster in clusters:
            hsv = np.average(clusters[cluster], axis=0) / 255
            rgb = convert_to_rgb(hsv)
            css = rgb_to_css(rgb)
            hexcolor = rgb_to_hex(rgb)
            polar = convert_to_polar(hsv[:-1])
            count = len(clusters[cluster])

            coverage = count / max
            add_to_palette = False
            candidate = {
                "value": rgb,
                "rgb": rgb_to_css(rgb),
                "hex": rgb_to_hex(rgb),
                "count": count,
            }

            if len(palette) < n or coverage >= cov_threshold:
                add_to_palette = True

            r_index = -1
            for other in palette:
                if get_distance(rgb, other["value"]) < dist_threshold:
                    if other["count"] >= count:
                        add_to_palette = False
                    else:
                        r_index = palette.index(other)
                    break

            if add_to_palette:
                if len(palette) >= n or r_index != -1:
                    palette.pop(r_index)
                palette.append(candidate)
                palette = sorted(palette, key=lambda x: x["count"], reverse=True)

            result[f"{cluster}"] = {
                "rgb": rgb,
                "css": css,
                "hex": hexcolor,
                "polar": polar,
                "count": count,
            }

        return result, palette
