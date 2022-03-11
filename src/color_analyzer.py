from sklearn.cluster import DBSCAN
from PIL import Image, ImageCms
import numpy as np
import colorsys
import math
import io


def hsv_to_polar(hsv):
    hue, sat, _ = np.array(hsv) / 255
    hue = hue * 2 * math.pi
    return (hue, sat)


def rgb_to_hsv(rgb):
    normalized = np.array(rgb) / 255
    rgb = np.round(np.multiply(colorsys.rgb_to_hsv(*normalized), 255))
    return tuple(int(x) for x in rgb)


def rgb_to_hex(rgb):
    return "%02x%02x%02x" % rgb


def rgb_to_lab(rgb):
    res = [*rgb]
    for i, value in enumerate(res):
        value /= 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        res[i] = value * 100

    r, g, b = res
    x, y, z = 0, 0, 0

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x /= 95.047
    y /= 100.0
    z /= 108.883

    res = [x, y, z]
    for i, value in enumerate(res):
        if value > 0.008856:
            res[i] = value ** (1 / 3)
        else:
            res[i] = (7.787 * value) + (16 / 116)

    x, y, z = res

    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return tuple(round(x) for x in [L, a, b])


def lab_to_rgb(lab):
    L, a, b = lab

    L *= 100 / 255
    a -= 127
    b -= 127

    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    res = [x, y, z]

    for i, value in enumerate(res):
        if value ** 3 > 0.008856:
            res[i] = value ** 3
        else:
            res[i] = (value - 16 / 116) / 7.787

    x, y, z = res

    x = (x * 95.047) / 100
    z = (z * 108.883) / 100

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    res = [r, g, b]
    for i, value in enumerate(res):
        if value > 0.0031308:
            res[i] = 1.055 * (value ** (1 / 2.4)) - 0.055
        else:
            res[i] *= 12.92
        res[i] *= 255

    return tuple(max(min(round(v), 255), 0) for v in res)


def get_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def open_image(bytes, max_size):
    image = Image.open(io.BytesIO(bytes))
    image.thumbnail(max_size)

    return np.array(image.convert("RGB").getdata())


class ColorAnalyzer:
    def __init__(self, image):
        self.max_size = (128, 128)
        self.epsilon = 0.1
        self.min_samples = 2

        self.image = open_image(image, self.max_size)

    def calculate_proportions(self):
        img = self.image

        colors, counts = np.unique(img, return_counts=True, axis=0)

        palette_size = 8
        dist_threshold = 30
        dist_weight = 0.5
        palette = []
        result = {}

        for i, cluster in enumerate(zip(colors, counts)):
            color, count = cluster
            if count == 1:
                continue

            count = int(count)
            rgb = tuple(int(x) for x in color.tolist())

            hexcolor = rgb_to_hex(rgb)

            hsv = rgb_to_hsv(rgb)
            polar = hsv_to_polar(hsv)

            lab = rgb_to_lab(rgb)

            add_to_palette = True
            candidate = {
                "lab": lab,
                "hsv": hsv,
                "hex": hexcolor,
                "count": count,
                "score": 0,
            }

            r_index = -1
            score = count + (palette_size - len(palette)) * 50 * dist_weight
            for other in palette:
                dist = get_distance(lab, other["lab"])
                score += dist * dist_weight
                if dist < dist_threshold:
                    r_index = palette.index(other)

            if r_index != -1 and palette[r_index]["score"] >= score:
                add_to_palette = False

            if add_to_palette:
                candidate["score"] = score
                if len(palette) >= palette_size or r_index != -1:
                    palette.pop(r_index)
                palette.append(candidate)
                palette = sorted(palette, key=lambda x: x["score"], reverse=True)

            result[f"{i}"] = {
                "rgb": rgb,
                "hex": hexcolor,
                "hsv": hsv,
                "polar": polar,
                "count": count,
            }

        palette = sorted(palette, key=lambda x: x["hsv"][0], reverse=True)

        return result, palette
