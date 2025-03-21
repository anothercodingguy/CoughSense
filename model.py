from transformers import TFWav2Vec2Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_audio

def build_model():
    # Load pre-trained Wav2Vec 2.0
    base_model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add classification head
    inputs = base_model.input
    x = base_model(inputs).last_hidden_state
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(3, activation='softmax')(x)
    
    return Model(inputs, outputs)

def train_model():
    # Load metadata
    metadata = pd.read_csv('coughvid-v2/metadata_compiled.csv')

    # Prepare labels
    le = LabelEncoder()
    labels = le.fit_transform(metadata['status'])  # COVID-19:0, healthy:1, pneumonia:2

    # Build and compile model
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train in batches
    batch_size = 32
    for i in range(0, len(metadata), batch_size):
        batch_files = metadata.iloc[i:i+batch_size]['file_name']
        batch_audio = [preprocess_audio(f) for f in batch_files]
        batch_labels = labels[i:i+batch_size]
        
        # Filter failed preprocess
        valid_samples = [(a,l) for a,l in zip(batch_audio, batch_labels) if a is not None]
        if not valid_samples:
            continue
            
        X = np.array([s[0] for s in valid_samples])
        y = np.array([s[1] for s in valid_samples])
        
        # Train
        model.train_on_batch(X, y)
        
    # Save the trained model
    model.save('coughsense.h5')
    print("Model saved as 'coughsense.h5'")

if __name__ == "__main__":
    train_model()