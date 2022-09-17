import os
import librosa
import numpy as np
import math
import json
import warnings
warnings.filterwarnings('ignore')

DATASET = "../../../Other/Datasets/genres/genres_original"
JSON_PATH = "data/genres.json"
SAMPLE_DURATION = 30


def generate_mfccs(dataset, json_path, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "genres": [],
        "mfccs": [],
        "targets": []
    }
    samples_per_segment = int((sample_rate * SAMPLE_DURATION) / num_segments)
    standard_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset)):
        if i > 0:
            folders = os.path.split(dirpath)
            label = folders[-1]
            data["genres"].append(label)
            print(f"\nProcessing {label}")
            for file in filenames:
                if not file.startswith(".DS_Store"):
                    path = os.path.join(dirpath, file)
                    signal, sr = librosa.load(path, sr=sample_rate)
                    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
                    for segment in range(num_segments):
                        initial_sample = samples_per_segment * segment
                        final_sample = initial_sample + samples_per_segment
                        mfcc = (librosa.feature.mfcc(
                            y=signal[initial_sample:final_sample], sr=sample_rate, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)).T
                        if len(mfcc) == standard_vectors_per_segment:
                            data["mfccs"].append(mfcc.tolist())
                            data["targets"].append(i - 1)
                            print(f"{path}, segment: {segment+1}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    generate_mfccs(DATASET, JSON_PATH, num_segments=10)
