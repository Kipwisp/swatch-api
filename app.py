from flask import Flask, request
from src.color_analyzer import ColorAnalyzer
from src.value_analyzer import ValueAnalyzer

app = Flask(__name__)


@app.route("/analyze", methods=['POST'])
def analyze_image():
    img = request.form['img']
    c_analyzer = ColorAnalyzer(img)
    v_analyzer = ValueAnalyzer(img)

    return {'color_proportion': c_analyzer.calculate_proportions(), 'value_distribution': v_analyzer.calculate_value_distribution()}
