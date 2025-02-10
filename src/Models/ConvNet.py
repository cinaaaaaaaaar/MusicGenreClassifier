import os, pickle, librosa, queue, math, warnings, sys, pickle, numpy as np, matplotlib.pyplot as plt
import logging

from random import randint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EpochProgressCallback(Callback):
    def __init__(self, epoch_queue, total_epochs):
        super().__init__()
        self.epoch_queue = epoch_queue
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Put the current epoch progress into the queue
        self.epoch_queue.put(f"{epoch+1}/{self.total_epochs}")


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
            self.inputs, self.targets, test_size=test_size
        )
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train, y_train, test_size=validation_size
        )
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
        convolutional_layers = [
            [32, (3, 3), (3, 3), (2, 2)],
            [64, (3, 3), (3, 3), (2, 2)],
            [128, (2, 2), (3, 3), (2, 2)],
        ]
        for i, layer in enumerate(convolutional_layers):
            if i == 0:
                model.add(
                    Conv2D(
                        layer[0], layer[1], activation="relu", input_shape=input_shape
                    )
                )
            else:
                model.add(Conv2D(layer[0], layer[1], activation="relu"))

            model.add(MaxPooling2D(layer[2], strides=layer[3], padding="same"))
            model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(128, activation="relu", kernel_regularizer=L2(0.5)))
        model.add(Dropout(0.8))
        model.add(Dense(len(self.genres), activation="softmax"))
        optimizer = Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model

    def train(self, batch_size, epochs, epoch_queue):
        total_epochs = epochs

        # Create the callback instance with the queue and total epochs
        epoch_callback = EpochProgressCallback(epoch_queue, total_epochs)

        # Start the training with the callback
        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_validation, self.y_validation),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[epoch_callback],  # Add the callback to the training process
        )

        # After training is finished, save weights and evaluate the model
        self.model.save_weights(self.weights_path)
        test_error, test_accuracy = self.model.evaluate(
            self.x_test, self.y_test, verbose=1
        )
        print(f"Accuracy on test set is {test_accuracy}")
        self.plot_history(history)

    def plot_history(self, history):
        """Schedules the plot to be shown on the main thread."""
        self.after(0, self._show_plot, history)

    def _show_plot(self, history):
        """Displays the plot on the main thread."""
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

        plt.show()  # Now this runs on the main thread

    def predict(
        self,
        audio_path,
        sample_rate,
        n_mfcc,
        n_fft,
        hop_length,
        samples_per_segment,
        overlap=0.5,
        segment_queue=None,
    ):

        self.model.load_weights(self.weights_path)
        signal, sr = librosa.load(audio_path, sr=sample_rate)

        step_size = int(samples_per_segment * (1 - overlap))
        standard_vectors_per_segment = self.x_train.shape[1]
        num_segments = max(1, int((len(signal) - samples_per_segment) / step_size) + 1)
        genre_scores = np.zeros(len(self.genres))

        logging.info(f"Processing file: {audio_path}")

        for segment in range(num_segments):
            initial_sample = step_size * segment
            final_sample = initial_sample + samples_per_segment

            if final_sample > len(
                signal
            ):  # Ensure segment does not exceed signal length
                break

            mfcc = librosa.feature.mfcc(
                y=signal[initial_sample:final_sample],
                sr=sample_rate,
                n_fft=n_fft,
                n_mfcc=n_mfcc,
                hop_length=hop_length,
            ).T

            if len(mfcc) == standard_vectors_per_segment:
                mfcc = mfcc[np.newaxis, ...]
                predictions = self.model.predict(mfcc)  # Get softmax probabilities
                genre_scores += predictions[0]  # Sum confidence scores across segments

                # Get the most confident genre for this segment
                segment_prediction_index = np.argmax(predictions[0])
                segment_genre = self.genres[segment_prediction_index]
                confidence_score = predictions[0][segment_prediction_index]

                segment_log = f"{segment+1}: {segment_genre.capitalize()}, Confidence: {confidence_score*100:.2f}%"
                logging.info(segment_log)

                if segment_queue:
                    segment_queue.put(segment_log)

        # Determine genre with the highest confidence score
        predicted_index = np.argmax(genre_scores)
        predicted_genre = self.genres[predicted_index]

        logging.info(f"Final Prediction: {predicted_genre}")

        return predicted_genre


def generate_mfccs(
    dataset_path,
    data_path,
    sample_rate,
    n_mfcc,
    n_fft,
    hop_length,
    num_segments,
    sample_duration,
    label_queue=None,
):
    data = {"genres": [], "inputs": [], "targets": []}
    samples_per_segment = int((sample_rate * sample_duration) / num_segments)
    standard_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if i > 0:
            folders = os.path.split(dirpath)
            label = folders[-1]
            data["genres"].append(label)

            if label_queue:
                label_queue.put(f"{label}")

            if i > 1:
                prev_label = data["genres"][i - 2]
                sys.stdout.write("\x1b[1A")
                sys.stdout.write("\x1b[2K")
                print(f"Processed {prev_label} ")
            print(f"Processing {label}...")

            for file in filenames:
                if not file.startswith(".DS_Store"):
                    path = os.path.join(dirpath, file)
                    signal, sr = librosa.load(path, sr=sample_rate)
                    for segment in range(num_segments):
                        initial_sample = samples_per_segment * segment
                        final_sample = initial_sample + samples_per_segment
                        mfcc = (
                            librosa.feature.mfcc(
                                y=signal[initial_sample:final_sample],
                                sr=sample_rate,
                                n_fft=n_fft,
                                n_mfcc=n_mfcc,
                                hop_length=hop_length,
                            )
                        ).T
                        if len(mfcc) == standard_vectors_per_segment:
                            data["inputs"].append(mfcc.tolist())
                            data["targets"].append(i - 1)
    if label_queue:
        label_queue.put("complete.")
    with open(data_path, "wb") as fp:
        pickle.dump(data, fp)
