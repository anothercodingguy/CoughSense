from transformers import TFWav2Vec2Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def build_model():
    base_model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    for layer in base_model.layers: 
        layer.trainable = False
    
    inputs = base_model.input
    x = base_model(inputs).last_hidden_state
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(3, activation='softmax')(x)
    return Model(inputs, outputs)