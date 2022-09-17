from Models.NeuralNetworks import GenreClassifier

DATA_PATH = "data/genres.json"
WEIGHTS_PATH = "weights/genres.h5"

model = GenreClassifier(DATA_PATH, WEIGHTS_PATH, 0.25, 0.2)
model.train(32, 30)
