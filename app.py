from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from preprocess import preprocess_audio
from model import build_model

app = Flask(__name__)
model = build_model()
model.load_weights('coughsense.h5')  # Pre-trained weights
classes = ['COVID-19', 'Healthy', 'Pneumonia']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        file = request.files['file']
        try:
            audio = preprocess_audio(file.stream)
            pred = model.predict(np.expand_dims(audio, axis=0))
            result = classes[np.argmax(pred)]
            confidence = f"{np.max(pred)*100:.1f}%"
            return render_template('result.html', 
                                 result=result,
                                 confidence=confidence)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))