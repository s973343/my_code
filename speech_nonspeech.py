import torch
import librosa

_silero_loaded = False
model = None
get_speech_timestamps = None
save_audio = None
read_audio = None
VADIterator = None
collect_chunks = None


def _load_silero_vad():
    global _silero_loaded, model
    global get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks

    if _silero_loaded:
        return

    # Explicitly trust the known repo to avoid torch.hub trust warnings.
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )

    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = utils
    _silero_loaded = True


def classify_audio(file_path, speech_threshold=0.2):
    """
    Classifies audio as Speech or Non-Speech.

    Args:
        file_path (str): Path to audio file
        speech_threshold (float): % of speech duration to classify as Speech

    Returns:
        dict: Results
    """

    _load_silero_vad()

    # Read audio (automatically resamples to 16k). On newer torchaudio builds
    # this may require torchcodec; if unavailable, fall back to librosa.
    try:
        wav = read_audio(file_path, sampling_rate=16000)
    except Exception as exc:
        print(
            f"[WARN] Silero read_audio failed ({type(exc).__name__}); "
            "falling back to librosa loader."
        )
        wav_np, _ = librosa.load(file_path, sr=16000, mono=True)
        wav = torch.from_numpy(wav_np).float()

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

    # Total duration
    total_duration = wav.shape[0] / 16000

    # Speech duration
    speech_duration = sum(
        (segment['end'] - segment['start']) for segment in speech_timestamps
    ) / 16000

    speech_ratio = speech_duration / total_duration

    classification = "Speech" if speech_ratio > speech_threshold else "Non-Speech"

    return {
        "total_duration_sec": round(total_duration, 2),
        "speech_duration_sec": round(speech_duration, 2),
        "speech_ratio": round(speech_ratio, 3),
        "classification": classification,
        "speech_segments": speech_timestamps
    }


# Example usage
if __name__ == "__main__":
    result = classify_audio("segment_001.wav")
    print(result)
