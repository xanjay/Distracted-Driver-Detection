import os

config = {
    "TRAIN_DIR": os.environ.get("TRAIN_DIR"),
    "TEST_DIR": os.environ.get("TEST_DIR"),
    "MODELS_DIR": "saved_models",
    "IMG_HEIGHT": 224,
    "IMG_WIDTH": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
}
