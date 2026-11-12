"""
Sample test of the YuNet face detector on ~40 representative images before
wiring it into the pipeline, per the "test on 30-50 images first" requirement.

Categories: real human faces (should detect), scenery/objects/patterns (should NOT).
"""

import glob
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

random.seed(42)

YUNET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_detection_yunet.onnx")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw",
                         "age_prediction_up", "age_prediction", "test")


def detect_yunet(pil_image, score_threshold=0.7):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    detector = cv2.FaceDetectorYN_create(YUNET_PATH, "", (w, h), score_threshold=score_threshold)
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)
    return [] if faces is None else faces


def make_synthetic_non_faces():
    """Objects/patterns that historically trip up Haar cascades: rings, textures, blobs."""
    imgs = {}

    ring = Image.new("RGB", (200, 200), (30, 30, 30))
    d = ImageDraw.Draw(ring)
    d.ellipse([40, 40, 160, 160], outline=(220, 190, 140), width=8)
    d.ellipse([90, 30, 110, 50], fill=(255, 255, 255))
    imgs["jewelry_ring"] = ring

    locket = Image.new("RGB", (200, 200), (20, 20, 25))
    d = ImageDraw.Draw(locket)
    d.ellipse([60, 70, 140, 150], fill=(200, 170, 90), outline=(255, 230, 150), width=4)
    d.line([100, 20, 100, 70], fill=(200, 170, 90), width=3)
    imgs["jewelry_locket"] = locket

    checker = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(checker)
    for i in range(0, 200, 25):
        for j in range(0, 200, 25):
            if (i + j) // 25 % 2 == 0:
                d.rectangle([i, j, i + 25, j + 25], fill=(0, 0, 0))
    imgs["checker_pattern"] = checker

    blank = Image.new("RGB", (200, 200), (128, 128, 128))
    imgs["blank_gray"] = blank

    noise = Image.fromarray((np.random.rand(200, 200, 3) * 255).astype("uint8"))
    imgs["random_noise"] = noise

    circles = Image.new("RGB", (200, 200), (245, 240, 230))
    d = ImageDraw.Draw(circles)
    d.ellipse([50, 60, 90, 100], fill=(80, 60, 40))
    d.ellipse([110, 60, 150, 100], fill=(80, 60, 40))
    d.ellipse([80, 120, 120, 140], fill=(150, 80, 80))
    imgs["face_like_blobs"] = circles

    return imgs


def main():
    results = []

    # Real faces - sample 20 from different age folders
    face_paths = []
    for age_dir in random.sample(sorted(os.listdir(DATA_DIR)), 20):
        full = os.path.join(DATA_DIR, age_dir)
        if os.path.isdir(full):
            files = os.listdir(full)
            if files:
                face_paths.append(os.path.join(full, random.choice(files)))

    for p in face_paths:
        img = Image.open(p)
        faces = detect_yunet(img)
        results.append(("REAL_FACE", os.path.basename(p), len(faces),
                         faces[0][-1] if len(faces) else None))

    # Non-face real photos (Windows wallpapers/scenery/screens)
    wallpapers = glob.glob(r"C:\Windows\Web\**\*.jpg", recursive=True)
    for p in random.sample(wallpapers, min(15, len(wallpapers))):
        img = Image.open(p)
        faces = detect_yunet(img)
        results.append(("NON_FACE_PHOTO", os.path.basename(p), len(faces),
                         faces[0][-1] if len(faces) else None))

    # Synthetic tricky non-faces (jewelry, patterns, blobs)
    for name, img in make_synthetic_non_faces().items():
        faces = detect_yunet(img)
        results.append(("SYNTHETIC_NON_FACE", name, len(faces),
                         faces[0][-1] if len(faces) else None))

    print(f"{'CATEGORY':<20} {'FILE':<40} {'N_FACES':<8} {'SCORE'}")
    print("-" * 80)
    tp = fn = tn = fp = 0
    for category, name, n, score in results:
        score_str = f"{score:.2f}" if score is not None else "-"
        print(f"{category:<20} {name[:38]:<40} {n:<8} {score_str}")
        if category == "REAL_FACE":
            if n >= 1:
                tp += 1
            else:
                fn += 1
        else:
            if n == 0:
                tn += 1
            else:
                fp += 1

    print("\n" + "=" * 40)
    print(f"Real faces detected:      {tp}/{tp + fn}")
    print(f"Non-faces correctly rejected: {tn}/{tn + fp}")
    print(f"False positives (non-face -> detected as face): {fp}")


if __name__ == "__main__":
    main()
