"""
Grad-CAM for the ResNet18 age regressor. Since this is regression (not classification),
the "class score" we backprop is just the model's scalar age output itself - a standard
adaptation of Grad-CAM for regression targets.

Usage: python ml/gradcam.py <image_path>  -> saves gradcam_overlay.png next to the input.
"""

import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from predict import get_device, load_model
from preprocessing import eval_transform

TARGET_LAYER_NAME = "layer4"  # last conv block in ResNet18 - standard Grad-CAM choice


def _get_target_layer(model):
    return dict(model.named_modules())[TARGET_LAYER_NAME]


def generate_gradcam(model, device, pil_image):
    """Returns a PIL.Image heatmap overlay the same size as pil_image, or None if
    Grad-CAM couldn't run cleanly (never blocks the core prediction)."""
    try:
        activations = {}
        gradients = {}

        target_layer = _get_target_layer(model)

        def fwd_hook(module, inp, out):
            activations["value"] = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0].detach()

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            tensor = eval_transform(pil_image).unsqueeze(0).to(device)
            tensor.requires_grad_(False)

            model.zero_grad()
            output = model(tensor)  # shape (1, 1), the raw predicted age
            output.squeeze().backward()

            act = activations["value"][0]      # (C, H, W)
            grad = gradients["value"][0]        # (C, H, W)

            weights = grad.mean(dim=(1, 2))     # (C,) - global average pooled gradients
            cam = torch.relu((weights[:, None, None] * act).sum(0))  # (H, W)

            cam = cam.cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()

            cam_resized = cv2.resize(cam, pil_image.size)
            heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            base = np.array(pil_image.convert("RGB").resize(pil_image.size))
            overlay = (0.55 * base + 0.45 * heatmap).astype(np.uint8)
            return Image.fromarray(overlay)
        finally:
            h1.remove()
            h2.remove()
    except Exception as e:
        print(f"Grad-CAM failed (non-fatal, skipping): {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ml/gradcam.py <image_path>")
        sys.exit(1)

    model, device = load_model()
    image = Image.open(sys.argv[1]).convert("RGB")
    overlay = generate_gradcam(model, device, image)
    if overlay is not None:
        out_path = sys.argv[1].rsplit(".", 1)[0] + "_gradcam.png"
        overlay.save(out_path)
        print(f"Saved {out_path}")
    else:
        print("Grad-CAM generation failed.")
