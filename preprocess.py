import json
from Models.ConvNet import generate_mfccs
with open("config.json", "r") as fp:
    config = json.load(fp)

DATASET = "../../../Other/Datasets/genres/genres_original"
DATA_PATH = "data/genres.pickle"
SAMPLE_DURATION = 30


if __name__ == "__main__":
    generate_mfccs(dataset_path=config["dataset_path"], data_path=config["data_path"],
                   num_segments=config["num_segments"], sample_duration=config["sample_duration"],
                   n_fft=config["n_fft"], sample_rate=config["sample_rate"], n_mfcc=config["n_mfcc"],
                   hop_length=config["hop_length"])
