from tensorflow.keras import layers, models, applications
import config

def create_ensemble_model(input_shape=(224, 224, 3), num_classes=101):
    """Ensemble model combining multiple architectures"""
    input_img = layers.Input(shape=input_shape)
    
    mobilenet_base = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    mobilenet_base.trainable = False
    
    mobilenet_out = mobilenet_base(input_img)
    mobilenet_out = layers.GlobalAveragePooling2D()(mobilenet_out)
    mobilenet_out = layers.Dense(256, activation='relu')(mobilenet_out)
    mobilenet_out = layers.Dropout(0.5)(mobilenet_out)
    
    efficientnet_base = applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    efficientnet_base.trainable = False
    
    efficientnet_out = efficientnet_base(input_img)
    efficientnet_out = layers.GlobalAveragePooling2D()(efficientnet_out)
    efficientnet_out = layers.Dense(256, activation='relu')(efficientnet_out)
    efficientnet_out = layers.Dropout(0.5)(efficientnet_out)
    
    cnn_stream = layers.Conv2D(64, (3, 3), activation='relu')(input_img)
    cnn_stream = layers.MaxPooling2D((2, 2))(cnn_stream)
    cnn_stream = layers.Conv2D(128, (3, 3), activation='relu')(cnn_stream)
    cnn_stream = layers.MaxPooling2D((2, 2))(cnn_stream)
    cnn_stream = layers.Conv2D(256, (3, 3), activation='relu')(cnn_stream)
    cnn_stream = layers.GlobalAveragePooling2D()(cnn_stream)
    cnn_stream = layers.Dense(256, activation='relu')(cnn_stream)
    cnn_stream = layers.Dropout(0.5)(cnn_stream)
    
    merged = layers.concatenate([mobilenet_out, efficientnet_out, cnn_stream])
    merged = layers.Dense(512, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    outputs = layers.Dense(num_classes, activation='softmax')(merged)
    
    model = models.Model(inputs=input_img, outputs=outputs)
    
    return model

