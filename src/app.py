import pandas as pd
import pickle
import joblib
from flask import Flask, render_template, request, send_file
from from_root import from_root
from sklearn.metrics._scorer import _SCORERS
import os

app = Flask(__name__)



# Load the pickled model
#with open(os.path.join(from_root(), "model/model.pkl"), 'rb') as file:
#    model = pickle.load(file)

with open(os.path.join(from_root(), "model/model_MLP.joblib"), 'rb') as file:
    model = joblib.load(file)

def classify_csv(file_path):
    df = pd.read_csv(file_path)
    # Assuming a column named 'text' for classification
    df['prediction'] = model.predict(df['text'])
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_csv_endpoint():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        df = classify_csv(file)
        output_file = 'classified_output.csv'
        df.to_csv(output_file, index=False)
        return send_file(output_file, as_attachment=True, download_name='classified_data.csv')

if __name__ == '__main__':
    app.run(debug=True)
