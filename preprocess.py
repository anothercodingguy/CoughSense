import librosa
import numpy as np

def preprocess_audio(file_stream):
    y, sr = librosa.load(file_stream, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    return librosa.util.normalize(y_trimmed)[:80000]  # 5-second clip