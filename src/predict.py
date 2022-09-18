import sys
from Models.ConvNet import Model
import json

with open("../config.json", "r") as fp:
    config = json.load(fp)

if len(sys.argv) < 2:
    print("Please enter a file path.")
else:
    audio_path = sys.argv[1]
    samples_per_segment = int(
        (config["sample_rate"] * config["sample_duration"]) / config["num_segments"])

    model = Model(data_path=config["data_path"], weights_path=config["weights_path"],
                  test_size=config["test_size"], validation_size=config["validation_size"])
    model.create_model()
    model.predict(audio_path, sample_rate=config["sample_rate"],
                  samples_per_segment=samples_per_segment, n_mfcc=config["n_mfcc"],
                  n_fft=config["n_fft"], hop_length=config["hop_length"])
