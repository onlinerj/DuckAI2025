from tensorflow.keras import layers, models, applications
import config

def create_two_stream_model(input_shape=(224, 224, 3), num_classes=101):
    """Two-stream CNN for spatial and temporal features"""
    spatial_input = layers.Input(shape=input_shape, name='spatial_input')
    
    spatial_base = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    spatial_base.trainable = False
    
    spatial_features = spatial_base(spatial_input)
    spatial_features = layers.GlobalAveragePooling2D()(spatial_features)
    spatial_features = layers.Dense(256, activation='relu')(spatial_features)
    spatial_features = layers.Dropout(0.5)(spatial_features)
    
    temporal_input = layers.Input(shape=input_shape, name='temporal_input')
    
    temporal_stream = layers.Conv2D(32, (3, 3), activation='relu')(temporal_input)
    temporal_stream = layers.MaxPooling2D((2, 2))(temporal_stream)
    temporal_stream = layers.Conv2D(64, (3, 3), activation='relu')(temporal_stream)
    temporal_stream = layers.MaxPooling2D((2, 2))(temporal_stream)
    temporal_stream = layers.Conv2D(128, (3, 3), activation='relu')(temporal_stream)
    temporal_stream = layers.GlobalAveragePooling2D()(temporal_stream)
    temporal_stream = layers.Dense(256, activation='relu')(temporal_stream)
    temporal_stream = layers.Dropout(0.5)(temporal_stream)
    
    merged = layers.concatenate([spatial_features, temporal_stream])
    merged = layers.Dense(512, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    outputs = layers.Dense(num_classes, activation='softmax')(merged)
    
    model = models.Model(inputs=[spatial_input, temporal_input], outputs=outputs)
    
    return model

