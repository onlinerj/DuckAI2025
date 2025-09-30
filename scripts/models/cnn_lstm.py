from tensorflow.keras import layers, models
import config

def create_cnn_lstm_model(sequence_length=5, input_shape=(224, 224, 3), num_classes=101):
    """CNN-LSTM model for temporal action recognition"""
    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])
    
    model = models.Sequential([
        layers.TimeDistributed(cnn_model, input_shape=(sequence_length, *input_shape)),
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

