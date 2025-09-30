from tensorflow.keras import layers, models, applications
import config

def create_mobilenet_finetuned_model(input_shape=(224, 224, 3), num_classes=101, 
                                     unfreeze_layers=50):
    """Fine-tuned MobileNetV2 model"""
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

