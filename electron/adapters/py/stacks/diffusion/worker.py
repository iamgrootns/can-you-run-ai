"""
Diffusers text-to-image worker. Spawned by electron/adapters/pydiffusion via
pyhost.runWorker: reads one JSON line from stdin (the job), streams NDJSON
events on stdout for the rest of its life, then exits.

This is the one-shot-process pattern (one worker per generation, same as
sd.cpp's adapter), not a persistent server — simplest thing that works, and
matches how the native adapters are already structured.
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
    except Exception as e:
        emit({"type": "error", "error": f"Python deps not available: {e}"})
        sys.exit(1)

    emit({"type": "hello"})

    model_path = job["modelPath"]
    prompt = job["prompt"]
    negative_prompt = job.get("negativePrompt") or None
    width = int(job.get("width") or 512)
    height = int(job.get("height") or 512)
    steps = int(job.get("steps") or 20)
    cfg_scale = float(job.get("cfgScale") if job.get("cfgScale") is not None else 7.0)
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
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, safety_checker=None)
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

    emit({"type": "progress", "text": "Generating..."})
    try:
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
        )
        try:
            result = pipe(**kwargs, callback_on_step_end=on_step)
        except TypeError:
            # Older diffusers versions use callback/callback_steps instead of
            # callback_on_step_end — fall back rather than hard-require the
            # newer signature.
            def legacy_cb(step, timestep, latents):
                emit({"type": "progress", "step": step + 1, "total": steps, "text": f"step {step + 1}/{steps}"})

            result = pipe(**kwargs, callback=legacy_cb, callback_steps=1)
        image = result.images[0]
        image.save(output_path)
    except Exception as e:
        emit({"type": "error", "error": f"Generation failed: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    emit({"type": "result", "outputPath": output_path})


if __name__ == "__main__":
    main()
