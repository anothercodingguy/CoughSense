import librosa
import numpy as np

def preprocess_audio(file_path):
    try:
        # Load audio directly from Kaggle
        y, sr = librosa.load(f'coughvid-v2/{file_path}', sr=16000)
        
        # Trim silence and normalize
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        y_normalized = librosa.util.normalize(y_trimmed)
        
        # Ensure 5-second length
        if len(y_normalized) < 80000:
            y_padded = np.pad(y_normalized, (0, 80000 - len(y_normalized)))
        else:
            y_padded = y_normalized[:80000]
            
        return y_padded
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None