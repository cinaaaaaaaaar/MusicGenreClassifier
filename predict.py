from Models.NeuralNetworks import GenreClassifier

DATA_PATH = "data/genres.json"
WEIGHTS_PATH = "weights/genres.h5"
audio_path = "../../../Other/datasets/genres/genres_original/metal/metal.00075.wav"

model = GenreClassifier(DATA_PATH, WEIGHTS_PATH, 0.25, 0.2)
model.model.load_weights(WEIGHTS_PATH)
model.predict(random=True)
