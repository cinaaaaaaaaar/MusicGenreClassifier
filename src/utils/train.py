from ..Models.ConvNet import Model
import json
import os

config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

with open(os.path.join(config_dir, "config.json"), "r") as f:
    config = json.load(f)
DATA_PATH = "data/genres.pickle"
WEIGHTS_PATH = "weights/genres.h5"

model = Model(
    data_path=config["data_path"],
    weights_path=config["weights_path"],
    test_size=config["test_size"],
    validation_size=config["validation_size"],
)
model.create_model()
model.train(batch_size=32, epochs=50)
