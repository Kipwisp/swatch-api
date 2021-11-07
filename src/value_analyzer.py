from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import io

def open_image(bytes, max_size):
    image = Image.open(io.BytesIO(bytes))
    image.thumbnail(max_size)

    return np.array(image.convert('L').getdata()) / 255


class ValueAnalyzer:
    def __init__(self, image):
        self.max_size = (256, 256)
        self.bins = 20

        self.image = open_image(image, self.max_size)

    def calculate_value_distribution(self):
        img = self.image

        counts, bins = np.histogram(img, bins=self.bins, range=(0.0, 1.0))

        result = {
            'counts': counts,
            'bins': bins[:-1]
        }

        return result

    def get_value_shapes(self):
        pass

    def show_plot(self, data):
        print(data)
        plt.plot(data['bins'], data['counts'])
        plt.show()



# im = Image.open(img)
# buf = io.BytesIO()
# im.save(buf, format='PNG')

# analyzer = ValueAnalyzer(buf.getvalue())
# result = analyzer.calculate_value_distribution()
# analyzer.show_plot(result)
