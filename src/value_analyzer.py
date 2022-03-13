from PIL import Image
import numpy as np
import io


class ValueAnalyzer:
    def __init__(self, image, max_size):
        self.max_size = max_size
        self.bins = 20

        self.image = self.open_image(image, self.max_size)

    def open_image(self, bytes, max_size):
        image = Image.open(io.BytesIO(bytes))
        image.thumbnail(max_size)

        return (np.array(image.convert("L").getdata()) / 255) * 100

    def calculate_value_distribution(self):
        img = self.image

        counts, bins = np.histogram(
            img,
            bins=20,
            range=(-3, 103),
            density=True,
        )

        bins = np.round(0.5 * (bins[1:] + bins[:-1]))
        counts = counts.tolist()
        bins = bins.tolist()
        result = {i: {"count": counts[i], "bin": bins[i]} for i in range(len(counts))}

        return result
