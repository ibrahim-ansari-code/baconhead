"""
Report what we see: describe a screen frame with a vision model (BLIP).
Lazy-loads the model on first use.
"""

from typing import Optional

import numpy as np

# Lazy singleton
_reporter = None


def get_reporter():
    """Load and return the BLIP captioner (lazy, once)."""
    global _reporter
    if _reporter is not None:
        return _reporter
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_id = "Salesforce/blip-image-captioning-base"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    _reporter = (processor, model, device)
    return _reporter


def describe_frame(frame: np.ndarray, max_new_tokens: int = 50) -> str:
    """
    Describe what's in the frame (e.g. gameplay screen).
    frame: RGB numpy array (H, W, 3), any size (will be resized by the model).
    """
    from PIL import Image

    processor, model, device = get_reporter()
    # BLIP expects PIL; resize if huge to avoid OOM / slowness
    pil = Image.fromarray(frame.astype(np.uint8))
    w, h = pil.size
    if max(w, h) > 384:
        pil = pil.resize((int(w * 384 / max(w, h)), int(h * 384 / max(w, h))), Image.Resampling.LANCZOS)
    import torch
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True).strip()
