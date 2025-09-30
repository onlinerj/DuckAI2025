import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import config
import utils

def get_callbacks(model_name, patience=None):
    """Get training callbacks"""
    if patience is None:
        patience = config.PATIENCE
    
    checkpoint_path = utils.get_model_checkpoint_path(model_name)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    return [early_stopping, model_checkpoint, reduce_lr]

def compile_model(model, learning_rate=None, num_classes=101):
    """Compile model with optimizer and loss"""
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, train_images, train_labels, val_images, val_labels,
                model_name="model", epochs=None, batch_size=None):
    """Train a model with given data"""
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    utils.log_message(f"Starting training: {model_name}")
    utils.log_message(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    callbacks = get_callbacks(model_name)
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    utils.log_message(f"Training complete: {model_name}")
    
    return history

def train_sequence_model(model, train_sequences, train_labels, val_sequences, val_labels,
                        model_name="model", epochs=None, batch_size=None):
    """Train a sequence-based model (LSTM, Conv3D)"""
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    utils.log_message(f"Starting sequence model training: {model_name}")
    utils.log_message(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    callbacks = get_callbacks(model_name)
    
    history = model.fit(
        train_sequences, train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    utils.log_message(f"Training complete: {model_name}")
    
    return history

def fine_tune_model(model, train_images, train_labels, val_images, val_labels,
                   base_model_name, unfreeze_layers=50, model_name="finetuned_model",
                   epochs=None, batch_size=None, learning_rate=1e-5):
    """Fine-tune a pretrained model"""
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    base_model = model.get_layer(base_model_name)
    base_model.trainable = True
    
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    utils.log_message(f"Fine-tuning: {model_name}")
    utils.log_message(f"Unfrozen layers: {unfreeze_layers}, LR: {learning_rate}")
    
    callbacks = get_callbacks(model_name)
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    utils.log_message(f"Fine-tuning complete: {model_name}")
    
    return history

