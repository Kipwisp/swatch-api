from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import io


def convert_to_polar(value):
    hue, sat = value / 255
    hue = (hue * 2 * math.pi)
    return (hue, sat)


def convert_to_rgb(value):
    value = value / 255
    return colorsys.hsv_to_rgb(*value)


def open_image(bytes, max_size):
    image = Image.open(io.BytesIO(bytes))
    image.thumbnail(max_size)

    return np.array(image.convert('HSV').getdata())


class ColorAnalyzer:
    def __init__(self, image):
        self.max_size = (256, 256)
        self.epsilon = 0.4
        self.min_samples = 2
        self.min_size = 20
        self.scale = 1000

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
            hsv = np.average(clusters[cluster], axis=0)
            rgb = convert_to_rgb(hsv)
            polar = convert_to_polar(hsv[:-1])
            count = len(clusters[cluster])

            result[cluster] = {'hsv': hsv, 'rgb': rgb, 'polar': polar, 'count': count}

        return result

    def show_plot(self, data):
        polar_coords = list(map(lambda x: data[x]['polar'], data.keys()))

        x = list(map(lambda x: x[0], polar_coords))
        y = list(map(lambda x: x[1], polar_coords))

        colors = list(map(lambda x: data[x]['rgb'], data.keys()))
        sizes = list(map(lambda x: data[x]['count'], data.keys()))

        max_size = max(sizes)
        sizes = list(map(lambda x: max((x / max_size) * self.scale, self.min_size), sizes))

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.scatter(x, y, c=colors, s=sizes)

        plt.show()


# im = Image.open(img)
# buf = io.BytesIO()
# im.save(buf, format='PNG')

# analyzer = ColorAnalyzer(buf.getvalue())
# result = analyzer.calculate_proportions()
# analyzer.show_plot(result)
