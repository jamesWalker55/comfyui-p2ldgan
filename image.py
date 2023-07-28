from PIL import Image

import numpy as np
import torch


def tensor_to_pil_image(tensor: torch.Tensor):
    assert len(tensor.shape) == 3
    return Image.fromarray(
        np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    )


def pil_image_to_tensor(image: Image.Image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)


def resize_image_to_resolution(image: Image.Image, resolution: int):
    longest_side = max(image.size)
    new_size = tuple(round(x / longest_side * resolution) for x in image.size)
    print(f"Resizing image from {image.size} to {new_size}")
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)
