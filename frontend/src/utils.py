"""Input validation for uploaded images. Keeps error messages user-friendly."""

from PIL import Image, UnidentifiedImageError

MAX_DIMENSION = 6000  # px, above this we consider the image "extremely large"
MIN_DIMENSION = 20    # px, below this the image is too small to be a usable photo
MAX_FILE_SIZE_MB = 15


def validate_uploaded_file(uploaded_file):
    """Returns (pil_image, error_message). Exactly one of them is None."""
    if uploaded_file is None:
        return None, "No file was uploaded."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return None, f"File is too large ({size_mb:.1f}MB). Max allowed is {MAX_FILE_SIZE_MB}MB."

    if uploaded_file.size == 0:
        return None, "The uploaded file is empty."

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None, "This file isn't a valid image (it may be corrupted or an unsupported format)."

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return None, f"Image is too small ({width}x{height}px). Please upload a larger photo."
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        return None, f"Image is extremely large ({width}x{height}px). Please upload a smaller photo."

    return image, None
