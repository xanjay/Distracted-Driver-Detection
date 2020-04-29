from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from src.utils import prepare_submission_df
from src.config import config
import os


MODELS_DIR = config["MODELS_DIR"]

IMG_HEIGHT = config['IMG_HEIGHT']
IMG_WIDTH = config['IMG_WIDTH']
BATCH_SIZE = config['BATCH_SIZE']

if __name__ == "__main__":
    model_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(config['TEST_DIR'], target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      color_mode='grayscale', classes=None, class_mode=None,
                                                      batch_size=BATCH_SIZE, shuffle=False)
    test_generator.reset()

    predictions = model.predict_generator(test_generator, steps=len(test_generator.filenames) / BATCH_SIZE)

    ids = [os.path.basename(p) for p in test_generator.filenames]
    final_df = prepare_submission_df(predictions, ids)

    final_df.to_csv("submission.csv", index=False)
