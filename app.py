from flask import Flask, request
from flask_cors import CORS
from src.color_analyzer import ColorAnalyzer
from src.value_analyzer import ValueAnalyzer
import json

app = Flask(__name__)
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze_image():
    img = request.data
    c_analyzer = ColorAnalyzer(img)
    v_analyzer = ValueAnalyzer(img)

    color_proportions = c_analyzer.calculate_proportions()

    response = {
        "color_proportion": color_proportions,
        "dominant_colors": c_analyzer.get_dominant_colors(color_proportions),
        "value_distribution": v_analyzer.calculate_value_distribution(),
    }

    print(response)
    return json.dumps(response)
