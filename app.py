from flask import Flask, request, render_template
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Load model
model = tf.keras.models.load_model('coughsense.h5')
class_names = ['COVID-19', 'Healthy', 'Pneumonia']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Empty filename')
        
        try:
            # Process without saving
            y, sr = librosa.load(file.stream, sr=16000)
            processed = preprocess_audio(y)
            
            # Predict
            pred = model.predict(np.expand_dims(processed, axis=0))
            result = class_names[np.argmax(pred)]
            confidence = float(np.max(pred))
            
            return render_template('result.html', 
                                result=result,
                                confidence=f"{confidence*100:.1f}%")
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)