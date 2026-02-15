import ffmpeg
import whisper
import os
import librosa
import soundfile as sf
from config import VIDEO_INPUT, AUDIO_OUTPUT, WHISPER_MODEL

def extract_and_transcribe(return_segments=False):
    print("-> Extracting and Transcribing Audio...")
    if not os.path.exists("data"):
        os.makedirs("data")

    # If there is no audio stream, skip all audio-related processing.
    try:
        probe = ffmpeg.probe(VIDEO_INPUT)
        has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    except ffmpeg.Error:
        has_audio = False

    if not has_audio:
        print("-> No audio stream found. Skipping audio pipeline.")
        if return_segments:
            return {"text": "", "segments": [], "formatted": []}
        return ""

    # Extract audio from video
    ffmpeg.input(VIDEO_INPUT).output(AUDIO_OUTPUT, acodec='libmp3lame').run(overwrite_output=True, quiet=False)

    # Transcribe with Whisper (speech-aware segments)
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        AUDIO_OUTPUT,
        language="hi",
        task="translate",
        fp16=False
    )

    # Segment and save speech chunks (optional side output)
    output_dir = "speech_segments"
    os.makedirs(output_dir, exist_ok=True)
    min_segment_duration = 2.0

    # Load audio waveform for slicing
    audio, sr = librosa.load(AUDIO_OUTPUT, sr=None, mono=True)

    merged_segments = []
    buffer_start = None
    buffer_end = None
    buffer_text = ""

    for seg in result.get("segments", []):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = (seg.get("text") or "").strip()

        if buffer_start is None:
            buffer_start = start
            buffer_end = end
            buffer_text = text
        else:
            buffer_end = end
            buffer_text += " " + text

        if (buffer_end - buffer_start) >= min_segment_duration:
            merged_segments.append({
                "start": buffer_start,
                "end": buffer_end,
                "text": buffer_text.strip()
            })
            buffer_start = None
            buffer_end = None
            buffer_text = ""

    if buffer_start is not None:
        merged_segments.append({
            "start": buffer_start,
            "end": buffer_end,
            "text": buffer_text.strip()
        })

    if merged_segments:
        print(f"-> Saving {len(merged_segments)} speech segments to '{output_dir}'...")

    formatted_segments = []
    for i, seg in enumerate(merged_segments, 1):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        duration = max(0.0, end - start)

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]

        out_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
        sf.write(out_path, segment_audio, sr)

        print(f"  Segment {i}: {start:.2f} - {end:.2f}s | {duration:.2f}s | {out_path}")
        if text:
            print(f"    Text: {text}")

        formatted_segments.append(
            f"Time     : {start:.2f} - {end:.2f} s\n"
            f"  Duration : {duration:.2f} s\n"
            f"  Text     : {text}"
        )

    transcript = result.get("text", "")
    if return_segments:
        return {
            "text": transcript,
            "segments": merged_segments,
            "formatted": formatted_segments
        }
    return transcript
