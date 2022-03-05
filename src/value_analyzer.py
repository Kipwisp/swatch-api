from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import io


def open_image(bytes, max_size):
    image = Image.open(io.BytesIO(bytes))
    image.thumbnail(max_size)

    return (np.array(image.convert("L").getdata()) / 255) * 100


class ValueAnalyzer:
    def __init__(self, image):
        self.max_size = (256, 256)
        self.bins = 20

        self.image = open_image(image, self.max_size)

    def calculate_value_distribution(self):
        img = self.image

        counts, bins = np.histogram(img, bins=self.bins, range=(0, 100), density=True)
        counts = counts.tolist()
        bins = bins[:-1].tolist()

        result = {i: {"count": counts[i], "bin": bins[i]} for i in range(len(counts))}

        print(result)

        return result

    def get_value_shapes(self):
        pass
