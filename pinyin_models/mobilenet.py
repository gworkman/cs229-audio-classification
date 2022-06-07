import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_pickle('../data/preprocessed/spectrogram.pkl').explode('spectrogram')
    x, y = np.array(df['spectrogram'].to_list()), pd.get_dummies(df['pinyin']).to_numpy()

    x = np.repeat(x[..., np.newaxis], 3, -1)
    n_classes = y.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, train_size=0.5, shuffle=True)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).cache()
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).cache()
    validation_ds = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).batch(32).cache()

    model = models.Sequential([
        layers.RandomCrop(128, 128),
        tf.keras.applications.MobileNetV2((128, 128, 3), include_top=False),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, 'softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=30
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(acc, label='Training Accuracy')
    ax[0].plot(val_acc, label='Validation Accuracy')
    ax[0].legend(loc='lower right')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim([min(ax[0].get_ylim()), 1])
    ax[0].set_title('Training and Validation Accuracy')

    ax[1].plot(loss, label='Training Loss')
    ax[1].plot(val_loss, label='Validation Loss')
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel('Cross Entropy')
    ax[1].set_ylim([0, 1.0])
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('epoch')
    plt.suptitle('MobileNetV2 Training Performance (Pronunciations)')
    plt.savefig('../images/pinyin_mobilenet_training.png')

    y_hat = tf.argmax(model.predict(test_ds), axis=1)
    test_accuracy = np.mean(y_hat == np.argmax(y_test, axis=1))

    fig, ax = plt.subplots()
    cm = ConfusionMatrixDisplay.from_predictions(np.argmax(y_test, axis=1), y_hat, ax=ax, include_values=False)
    plt.title(f'Confusion for MobileNetV2 on Mel Spectrogram features (test accuracy: {test_accuracy:0.2f})')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    plt.savefig(f'../images/mobilenet_confusion_pinyin.png')

    model.save('./mobilenet-pinyin')
