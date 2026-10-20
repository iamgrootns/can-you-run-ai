"""
Diffusers text-to-video worker. Same one-shot-process shape and NDJSON
protocol as the image worker (electron/adapters/py/stacks/diffusion) — video
pipelines in diffusers use the same DiffusionPipeline.from_pretrained +
callback_on_step_end API, just returning frames instead of a single image,
which export_to_video then muxes into an mp4.
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
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
    except Exception as e:
        emit({"type": "error", "error": f"Python deps not available: {e}"})
        sys.exit(1)

    emit({"type": "hello"})

    model_path = job["modelPath"]
    prompt = job["prompt"]
    negative_prompt = job.get("negativePrompt") or None
    num_frames = int(job.get("numFrames") or 16)
    fps = int(job.get("fps") or 8)
    steps = int(job.get("steps") or 25)
    seed = job.get("seed")
    output_path = job["outputPath"]
    requested_device = job.get("device") or "cpu"

    if requested_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif requested_device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    emit({"type": "progress", "text": f"Loading pipeline on {device} ({dtype})..."})
    try:
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe = pipe.to(device)
    except Exception as e:
        emit({"type": "error", "error": f"Failed to load pipeline: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    generator = None
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    def on_step(pipe_, step, timestep, callback_kwargs):
        emit({"type": "progress", "step": step + 1, "total": steps, "text": f"step {step + 1}/{steps}"})
        return callback_kwargs

    emit({"type": "progress", "text": f"Generating {num_frames} frames..."})
    try:
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=steps,
            generator=generator,
        )
        try:
            result = pipe(**kwargs, callback_on_step_end=on_step)
        except TypeError:
            def legacy_cb(step, timestep, latents):
                emit({"type": "progress", "step": step + 1, "total": steps, "text": f"step {step + 1}/{steps}"})

            result = pipe(**kwargs, callback=legacy_cb, callback_steps=1)
        frames = result.frames[0] if hasattr(result, "frames") else result.images
        export_to_video(frames, output_path, fps=fps)
    except Exception as e:
        emit({"type": "error", "error": f"Generation failed: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    emit({"type": "result", "outputPath": output_path})


if __name__ == "__main__":
    main()
