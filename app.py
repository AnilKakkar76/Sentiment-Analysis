import io
import pandas as pd
from flask import Flask, render_template, request, jsonify
from analyzer import SentimentAnalyzer

app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    return jsonify(analyzer.predict(text))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'GET':
        return render_template('dashboard.html', summary=None)

    file = request.files.get('csv')
    if not file:
        return render_template('dashboard.html', summary=None, error="Upload a CSV file with a 'text' column.")

    df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
    if 'text' not in df.columns:
        return render_template('dashboard.html', summary=None, error="CSV must have a 'text' column.")

    results = [analyzer.predict(t) for t in df['text'].astype(str).tolist()]
    counts = {'positive': 0, 'neutral': 0, 'negative': 0, 'uncertain': 0}
    for r in results:
        counts[r['label']] = counts.get(r['label'], 0) + 1

    return render_template('dashboard.html', summary=counts, total=len(results))

if __name__ == '__main__':
    app.run(debug=True, port=5000)