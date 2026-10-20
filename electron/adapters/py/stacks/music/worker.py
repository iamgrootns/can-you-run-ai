"""
MusicGen-family text-to-music worker. Same one-shot-process shape as the
diffusion worker: one JSON line in on stdin, NDJSON events out on stdout.
Uses transformers directly (no diffusers) — MusicGen/AudioGen-style models
are AutoProcessor + a ConditionalGeneration causal LM, not a diffusion U-Net.
"""
import sys
import json
import traceback


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    line = sys.stdin.readline()
    job = json.loads(line)

    try:
        import torch
        import scipy.io.wavfile
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except Exception as e:
        emit({"type": "error", "error": f"Python deps not available: {e}"})
        sys.exit(1)

    emit({"type": "hello"})

    model_path = job["modelPath"]
    prompt = job["prompt"]
    duration_seconds = float(job.get("durationSeconds") or 8.0)
    output_path = job["outputPath"]
    requested_device = job.get("device") or "cpu"

    if requested_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif requested_device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    emit({"type": "progress", "text": f"Loading model on {device}..."})
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        model = MusicgenForConditionalGeneration.from_pretrained(model_path).to(device)
    except Exception as e:
        emit({"type": "error", "error": f"Failed to load model: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    # MusicGen's own frame rate — ~50 generated tokens per second of audio,
    # per the model card / generation config. Real figure comes from the
    # loaded model's config, not a hardcoded guess, so this scales correctly
    # for the tiny test fixture (tiny token rate) and the real models alike.
    frame_rate = getattr(model.config.audio_encoder, "frame_rate", 50)
    max_new_tokens = int(duration_seconds * frame_rate)

    emit({"type": "progress", "text": f"Generating ~{duration_seconds}s of audio ({max_new_tokens} tokens)..."})
    try:
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio = audio_values[0, 0].detach().cpu().numpy()
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio)
    except Exception as e:
        emit({"type": "error", "error": f"Generation failed: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    emit({"type": "result", "outputPath": output_path})


if __name__ == "__main__":
    main()
