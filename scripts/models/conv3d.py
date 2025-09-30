from tensorflow.keras import layers, models
import config

def create_conv3d_model(sequence_length=5, input_shape=(224, 224, 3), num_classes=101):
    """3D Convolutional model for spatio-temporal action recognition"""
    model = models.Sequential([
        layers.Conv3D(32, (3, 3, 3), activation='relu', 
                     input_shape=(sequence_length, *input_shape)),
        layers.MaxPooling3D((2, 2, 2)),
        
        layers.Conv3D(64, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        
        layers.Conv3D(128, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        
        layers.Conv3D(256, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

