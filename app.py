from flask import Flask, request
from src.color_analyzer import ColorAnalyzer
from src.value_analyzer import ValueAnalyzer
import json

app = Flask(__name__)


@app.route("/analyze", methods=['POST'])
def analyze_image():
    img = request.data
    c_analyzer = ColorAnalyzer(img)
    v_analyzer = ValueAnalyzer(img)

    print('testing', flush=True)

    return json.dumps({'color_proportion': c_analyzer.calculate_proportions(), 'value_distribution': v_analyzer.calculate_value_distribution()})
