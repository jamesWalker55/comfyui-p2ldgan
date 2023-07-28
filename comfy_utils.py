import os
from pathlib import Path

import comfy.model_management as model_management
import folder_paths

# This is the 'comfyui-p2ldgan' folder
BASE_DIR = Path(__file__).absolute().parent

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

folder_paths.folder_names_and_paths["p2ldgan"] = (
    [str(BASE_DIR / "checkpoints")],
    {".pt"},
)


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def get_p2ldgan_model_path() -> Path:
    search_paths = folder_paths.get_folder_paths("p2ldgan")

    for search_path in search_paths:
        model_path = Path(search_path) / "p2ldgan_generator_200.pth"
        print("Finding", model_path)
        if model_path.is_symlink():
            model_path = Path(os.path.join(model_path.parent, os.readlink(model_path)))
            print("Symlink, replace", model_path)
        if model_path.exists():
            print("Exists!")
            return model_path
        print("Doesn't exist")

    raise FileNotFoundError(
        "Failed to find 'p2ldgan_generator_200.pth', please place it here: ComfyUI/custom_nodes/comfyui-p2ldgan/checkpoints"
    )
