# feature_audio.py
import librosa
import numpy as np
import soundfile as sf


def extract_prosodic(path, sr=16000):
    """
    Extract lightweight prosodic features from audio:
    - energy
    - pause ratio
    - onset/speaking rate
    """
    # Load audio
    y, _ = librosa.load(path, sr=sr)
    duration = len(y) / sr

    if duration == 0:
        return {
            "energy": 0.0,
            "pause_ratio": 0.0,
            "onset_rate": 0.0
        }

    # RMS Energy
    energy = float(np.mean(librosa.feature.rms(y=y)))

    # Pause ratio (non-speech percentage)
    intervals = librosa.effects.split(y, top_db=30)
    total_speech = sum([(e - s) for s, e in intervals]) / sr
    pause_ratio = float(1 - (total_speech / duration))

    # Onset/speaking rate
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = float(len(onsets) / duration)

    return {
        "energy": energy,
        "pause_ratio": pause_ratio,
        "onset_rate": onset_rate
    }


if __name__ == "__main__":
    print("✔ Audio feature extractor ready.")
