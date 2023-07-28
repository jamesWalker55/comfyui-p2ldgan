from comfy_utils import register_node


@register_node("TestNode", "Test Node")
class TestNode:
    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "image_dir": ("STRING", {"default": "./images", "multiline": False}),
            "glob_pattern": ("STRING", {"default": "*.png", "multiline": False}),
        }
    }

    RETURN_NAMES = ("IMAGE", "FRAME_COUNT")
    RETURN_TYPES = ("IMAGE", "INT")

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(self, image_dir: str, glob_pattern: str):
        pass
