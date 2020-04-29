import os
import logging
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from src.model_builder import ModelBuilder  # build_rmsprop_model21, build_resnetlike_model
from src.utils import plot_model_loss
from src.config import config


IMG_HEIGHT = config['IMG_HEIGHT']
IMG_WIDTH = config['IMG_WIDTH']
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']

MODELS_DIR = config["MODELS_DIR"]

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Setup Callbacks:
    filepath = os.path.join(MODELS_DIR, 'epoch{epoch:02d}-loss{loss:.2f}-val_loss{val_loss:.2f}.hdf5')
    # checkpoint
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                       save_best_only=True, save_weights_only=False, mode='min', period=1)

    # early stopping: patience = epochs
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1,
                                   mode='min', baseline=None, restore_best_weights=True)

    # Instantiate model
    model = ModelBuilder().build_resnetlike_model()

    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        color_mode='grayscale', classes=None, class_mode='categorical',
                                                        batch_size=BATCH_SIZE, shuffle=True, subset='training')
    valid_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        color_mode='grayscale', classes=None, class_mode='categorical',
                                                        batch_size=BATCH_SIZE, shuffle=True, subset='validation')
    # Train model:
    history_v1 = model.fit_generator(train_generator,
                                     steps_per_epoch=train_generator.samples // BATCH_SIZE,
                                     epochs=EPOCHS,
                                     callbacks=[early_stopping, model_checkpoint],
                                     verbose=1,
                                     validation_data=valid_generator,
                                     validation_steps=valid_generator.samples // BATCH_SIZE)

    logging.info("Finished Training.")
    model_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
    model.save(model_path)
    logging.info(model_path)
    plot_model_loss(history_v1.history)
