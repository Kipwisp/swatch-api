from flask import Flask, request
from flask_cors import CORS
from src.color_analyzer import ColorAnalyzer
from src.value_analyzer import ValueAnalyzer
import json

app = Flask(__name__)
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze_image():
    max_size = (150, 150)

    img = request.data
    c_analyzer = ColorAnalyzer(img, max_size)
    v_analyzer = ValueAnalyzer(img, max_size)

    color_proportions, palette = c_analyzer.calculate_proportions()
    value_distribution = v_analyzer.calculate_value_distribution()

    response = {
        "color_proportion": color_proportions,
        "color_palette": palette,
        "value_distribution": value_distribution,
    }

    return json.dumps(response)
