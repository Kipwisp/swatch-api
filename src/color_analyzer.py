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
        if value**3 > 0.008856:
            res[i] = value**3
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
    return np.linalg.norm(x - y)


class ColorAnalyzer:
    def __init__(self, image, max_size):
        self.max_size = max_size
        self.image, self.width, self.height = self.open_image(image, self.max_size)

    def open_image(self, bytes, max_size):
        rgb2lab = ImageCms.buildTransformFromOpenProfiles(
            ImageCms.createProfile("sRGB"),
            ImageCms.createProfile("LAB"),
            "RGB",
            "LAB",
        )
        image = Image.open(io.BytesIO(bytes))
        image.thumbnail(max_size)

        return (
            np.array(ImageCms.applyTransform(image.convert("RGB"), rgb2lab).getdata()),
            image.width,
            image.height,
        )

    def cluster_colors(self, img):
        sample_step = 10
        dist_threshold = 20

        clusters = {}
        for i in range(0, len(img), sample_step):
            pixel = img[i]
            found_cluster = False
            for cluster in clusters:
                if get_distance(cluster, pixel) < dist_threshold:
                    clusters[cluster]["values"].append(pixel)
                    found_cluster = True
                    break

            if not found_cluster:
                clusters[tuple(pixel.tolist())] = {"values": [pixel], "index": i}

        return clusters

    def calculate_proportions(self):
        img = self.image

        clusters = self.cluster_colors(img)

        palette_size = 8
        dist_threshold = 35
        dist_weight = 0.7
        count_weight = 0.3
        bias = 25
        palette = []
        result = {}

        for i, cluster in enumerate(clusters):
            colors = clusters[cluster]["values"]
            index = clusters[cluster]["index"]
            count = len(colors)

            if count == 1:
                continue

            lab = np.average(colors, axis=0)

            rgb = lab_to_rgb(lab)
            hexcolor = rgb_to_hex(rgb)
            hsv = rgb_to_hsv(rgb)
            polar = hsv_to_polar(hsv)

            row = int(index // self.width)
            column = int(index % self.width)
            pos = (column, row)

            add_to_palette = True
            candidate = {
                "lab": lab,
                "hsv": hsv,
                "hex": hexcolor,
                "count": count,
                "pos": pos,
                "score": 0,
            }

            violations = 0
            r_index = -1
            score = (
                count * count_weight
                + (palette_size - len(palette)) * bias * dist_weight
            )
            for j, other in enumerate(palette):
                dist = get_distance(lab, other["lab"])
                score += dist * dist_weight
                if dist < dist_threshold:
                    r_index = j
                    violations += 1

                if violations > 1:
                    add_to_palette = False
                    break

            if r_index != -1 and palette[r_index]["score"] >= score:
                add_to_palette = False

            if add_to_palette:
                candidate["score"] = score
                if len(palette) >= palette_size or r_index != -1:
                    r_index = (
                        min(range(len(palette)), key=palette.__getitem__["score"])
                        if r_index == -1
                        else r_index
                    )
                    palette.pop(r_index)

                palette.append(candidate)

            result[f"{i}"] = {
                "rgb": rgb,
                "hex": hexcolor,
                "hsv": hsv,
                "polar": polar,
                "count": count,
                "pos": pos,
            }

        palette = sorted(palette, key=lambda x: x["hsv"][0], reverse=True)
        palette = [{"hex": x["hex"], "pos": x["pos"]} for x in palette]

        return result, palette
