from tensorflow.keras import layers, models, applications
import config

def create_pretrained_cnn_lstm_model(sequence_length=5, input_shape=(224, 224, 3), 
                                    num_classes=101):
    """Pretrained CNN-LSTM model using MobileNetV2"""
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    cnn_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D()
    ])
    
    model = models.Sequential([
        layers.TimeDistributed(cnn_model, input_shape=(sequence_length, *input_shape)),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(128),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

