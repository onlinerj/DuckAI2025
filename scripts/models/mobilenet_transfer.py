from tensorflow.keras import layers, models, applications
import config

def create_mobilenet_transfer_model(input_shape=(224, 224, 3), num_classes=101):
    """MobileNetV2 transfer learning model"""
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

