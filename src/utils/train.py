from Models.ConvNet import Model
import json

with open("/Users/kisisel/Desktop/Kodlama/Python/Machine Learning/genre_classification_tf/config.json", "r") as fp:
    config = json.load(fp)
DATA_PATH = "data/genres.pickle"
WEIGHTS_PATH = "weights/genres.h5"

model = Model(data_path=config["data_path"], weights_path=config["weights_path"],
              test_size=config["test_size"], validation_size=config["validation_size"])
model.create_model()
model.train(batch_size=32, epochs=50)
