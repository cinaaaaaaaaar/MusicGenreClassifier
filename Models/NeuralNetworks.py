import json
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from random import randint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GenreClassifier():
    def __init__(self, data_path, weights_path, test_size, validation_size):
        self.data_path = data_path
        self.weights_path = weights_path
        self.load_data(data_path)
        self.train_test_validation_split(test_size, validation_size)
        self.create_model()

    def load_data(self, data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)

        inputs = np.array(data["mfccs"])
        targets = np.array(data["targets"])
        self.genres = genres = data["genres"]
        self.inputs = inputs
        self.targets = targets
        return inputs, targets

    def train_test_validation_split(self, test_size, validation_size):
        x_train, x_test, y_train, y_test = train_test_split(
            self.inputs, self.targets, test_size=test_size)
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train, y_train, test_size=validation_size)
        x_train = x_train[..., np.newaxis]
        x_validation = x_validation[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        self.x_train = x_train
        self.x_validation = x_validation
        self.x_test = x_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test

    def create_model(self):
        x_train = self.x_train
        model = Sequential()
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        convolutional_layers = [[32, (3, 3), (3, 3), (2, 2)], [32, (3, 3), (3, 3), (2, 2)],
                                [32, (2, 2), (3, 3), (2, 2)]]
        for i, layer in enumerate(convolutional_layers):
            if i == 0:
                model.add(Conv2D(
                    layer[0], layer[1], activation="relu", input_shape=input_shape))
            else:
                model.add(Conv2D(layer[0], layer[1], activation="relu"))

            model.add(MaxPooling2D(layer[2], strides=layer[3], padding='same'))
            model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))
        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train(self, batch_size, epochs):
        history = self.model.fit(self.x_train, self.y_train, validation_data=(
            self.x_validation, self.y_validation), batch_size=batch_size, epochs=epochs)
        self.model.save_weights(self.weights_path)
        test_error, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print(f"Accuracy on test set is {test_accuracy}")
        self.plot_history(history)

    def plot_history(self, history):
        fig, axs = plt.subplots(2)

        axs[0].plot(history.history["accuracy"], label="train accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy Eval")

        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error Eval")

        plt.show()

    def predict(self, index=None, random=True):
        if random:
            index = randint(0, len(self.inputs))
        x_test = self.inputs[index][np.newaxis, ...]

        prediction = self.model.predict(x_test)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_genre = self.genres[predicted_index[0]]
        expected_index = self.targets[index]
        expected_genre = self.genres[expected_index]
        print(f"Predicted genre: {predicted_genre}")
        print(f"Expected genre: {expected_genre}")
        print(f"MFCC index: {index}")
