# Distracted Driver Detection

![Distracted Driver Detection](https://storage.googleapis.com/kaggle-media/competitions/kaggle/5048/media/output_DEb8oT.gif)

Classifying driver distraction from dashcam images using a CNN built with Keras/TensorFlow.

## Data Source

- Kaggle competition dataset:  
  https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data
- Target format used by this project is 10 classes (safe driving, texting, phone calls, reaching, etc.).

## Approach

- **Input pipeline:** images are resized to **224x224**, converted to **grayscale**, and normalized with `rescale=1./255`.
- **Train/validation split:** `ImageDataGenerator(validation_split=0.2)` (80/20 split).
- **Model:** custom CNN with residual-style blocks (`build_resnetlike_model` in `src/model_builder.py`), trained with **RMSprop** and categorical cross-entropy.
- **Training callbacks:** early stopping and model checkpointing.
- **Inference output:** generates `submission.csv` with columns `c0...c9` and `img`.

## Results

- Validation accuracy: **97.0%**
- Kaggle leaderboard log-loss: **1.76805** (private test set)

## Run

From the repository root:

```bash
pip install -r requirements.txt

export TRAIN_DIR=/absolute/path/to/train
export TEST_DIR=/absolute/path/to/test

python -m src.train
python -m src.predict
```

## Notes

- Training saves the final model to `saved_models/cnn_model.h5`.
- This repository contains a baseline implementation and training/inference scripts for the competition format.
