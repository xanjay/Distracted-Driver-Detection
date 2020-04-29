import pandas as pd
# visualization
import matplotlib.pyplot as plt


def prepare_submission_df(predictions, ids):
    result_df = pd.DataFrame(predictions, columns=["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"])
    result_df['img'] = ids
    return result_df


def plot_model_loss(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
