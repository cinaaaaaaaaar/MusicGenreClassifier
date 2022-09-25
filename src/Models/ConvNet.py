import os, pickle, librosa, math, warnings, sys, pickle, numpy as np, matplotlib.pyplot as plt
from random import randint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:
    def __init__(self, data_path, weights_path, test_size, validation_size):
        self.weights_path = weights_path
        self.load_data(data_path, test_size, validation_size)

    def load_data(self, data_path, test_size, validation_size):
        with open(data_path, "rb") as fp:
            data = pickle.load(fp)

        inputs = np.array(data["inputs"])
        targets = np.array(data["targets"])
        self.genres = genres = data["genres"]
        self.inputs = inputs
        self.targets = targets
        self.train_test_validation_split(test_size, validation_size)

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
        convolutional_layers = [[32, (3, 3), (3, 3), (2, 2)],
                                [64, (3, 3), (3, 3), (2, 2)],
                                [128, (2, 2), (3, 3), (2, 2)]]
        for i, layer in enumerate(convolutional_layers):
            if i == 0:
                model.add(Conv2D(
                    layer[0], layer[1], activation="relu", input_shape=input_shape))
            else:
                model.add(Conv2D(layer[0], layer[1], activation="relu"))

            model.add(MaxPooling2D(layer[2], strides=layer[3], padding='same'))
            model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=L2(0.5)))
        model.add(Dropout(0.8))
        model.add(Dense(len(self.genres), activation='softmax'))
        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train(self, batch_size, epochs):
        history = self.model.fit(self.x_train, self.y_train, validation_data=(
            self.x_validation, self.y_validation), batch_size=batch_size, epochs=epochs)
        self.model.save_weights(self.weights_path)
        test_error, test_accuracy = self.model.evaluate(
            self.x_test, self.y_test, verbose=1)
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

    def predict(self, audio_path, sample_rate, n_mfcc, n_fft, hop_length, samples_per_segment):
        self.model.load_weights(self.weights_path)
        signal, sr = librosa.load(audio_path, sr=sample_rate)
        sample_duration = len(signal) / sr
        standard_vectors_per_segment = self.x_train.shape[1]
        num_segments = int((sample_duration * sample_rate) / samples_per_segment)
        predictions = []
        for segment in range(num_segments):
            initial_sample = samples_per_segment * segment
            final_sample = initial_sample + samples_per_segment
            mfcc = (librosa.feature.mfcc(
                y=signal[initial_sample:final_sample], sr=sample_rate,
                n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)).T
            if len(mfcc) == standard_vectors_per_segment:
                mfcc = mfcc[np.newaxis, ...]
                prediction = np.argmax(self.model.predict(mfcc), axis=1)[0]
                predictions.append(prediction)
        prediction_index = max(set(predictions), key=predictions.count)
        genre = self.genres[prediction_index]
        print(f"Predicted genre: {genre}")


def generate_mfccs(dataset_path, data_path, sample_rate, n_mfcc, n_fft,
                   hop_length, num_segments, sample_duration):
    data = {
        "genres": [],
        "inputs": [],
        "targets": []
    }
    samples_per_segment = int((sample_rate * sample_duration) / num_segments)
    standard_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if i > 0:
            folders = os.path.split(dirpath)
            label = folders[-1]
            data["genres"].append(label)
            if i > 1:
                prev_label = data["genres"][i - 2]
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
                print(f"Processed {prev_label} ")
            print(f"Processing {label}...")

            for file in filenames:
                if not file.startswith(".DS_Store"):
                    path = os.path.join(dirpath, file)
                    signal, sr = librosa.load(path, sr=sample_rate)
                    for segment in range(num_segments):
                        initial_sample = samples_per_segment * segment
                        final_sample = initial_sample + samples_per_segment
                        mfcc = (librosa.feature.mfcc(
                            y=signal[initial_sample:final_sample], sr=sample_rate,
                            n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)).T
                        if len(mfcc) == standard_vectors_per_segment:
                            data["inputs"].append(mfcc.tolist())
                            data["targets"].append(i - 1)
    with open(data_path, "wb") as fp:
        pickle.dump(data, fp)
