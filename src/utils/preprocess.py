import json
from ..Models.ConvNet import generate_mfccs
import os, queue

config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

with open(os.path.join(config_dir, "config.json"), "r") as f:
    config = json.load(f)

DATASET = "../../../../Other/Datasets/genres/genres_original"
DATA_PATH = "data/genres.pickle"
SAMPLE_DURATION = 30


def preprocess_data(label_queue=None):
    generate_mfccs(
        dataset_path=config["dataset_path"],
        data_path=config["data_path"],
        num_segments=config["num_segments"],
        sample_duration=config["sample_duration"],
        n_fft=config["n_fft"],
        sample_rate=config["sample_rate"],
        n_mfcc=config["n_mfcc"],
        hop_length=config["hop_length"],
        label_queue=label_queue,
    )


if __name__ == "__main__":
    preprocess_data()
