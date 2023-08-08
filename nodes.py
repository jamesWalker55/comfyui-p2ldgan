import torch
from torchvision import transforms

from .comfy_utils import get_p2ldgan_model_path, model_management, register_node
from .image import pil_image_to_tensor, resize_image_to_resolution, tensor_to_pil_image
from .models import Generator

generator: Generator | None = None


def load_generator():
    global generator

    if generator is not None:
        return generator

    try:
        model_path = get_p2ldgan_model_path()
        state_dict = torch.load(model_path)

        generator = Generator()

        # comfyui management, stolen from comfyui-animatediff
        offload_device = model_management.unet_offload_device()
        generator = generator.to(offload_device)

        generator.load_state_dict(state_dict)
        generator.eval()

        return generator
    except Exception as e:
        generator = None
        raise e from None


@register_node("P2LDGAN", "P2LDGAN")
class P2LDGANNode:
    CATEGORY = "jamesWalker55"

    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "resolution": ("INT", {"default": 1024, "min": 0, "step": 1}),
            "deep_sampling": (["yes", "no"],),
            "invert_output": (["yes", "no"],),
        }
    }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)

    OUTPUT_NODE = False

    FUNCTION = "execute"

    def execute(
        self,
        images: torch.Tensor,
        resolution: int,
        deep_sampling: str,
        invert_output: str,
    ):
        assert isinstance(images, torch.Tensor)
        assert isinstance(resolution, int)
        assert deep_sampling in ("yes", "no")
        assert invert_output in ("yes", "no")
        deep_sampling: bool = deep_sampling == "yes"
        invert_output: bool = invert_output == "yes"

        generator = load_generator()

        if resolution > 0:
            old_images = images
            images = []
            for img in old_images:
                img = tensor_to_pil_image(img)
                img = resize_image_to_resolution(img, resolution)
                img = pil_image_to_tensor(img)
                images.append(img)
            images = torch.stack(images)

        ld_images = self.generate_line_drawing(
            generator,
            images,
            deep_sampling=deep_sampling,
        )

        if invert_output:
            ld_images = ld_images * -1 + 1

        return (ld_images,)

    @staticmethod
    def generate_line_drawing(
        generator: Generator,
        images: torch.Tensor,
        deep_sampling: bool = True,
    ):
        # convert NHWC format to NCHW format
        images = images.permute(0, 3, 1, 2)

        # following code is derived from p2ldgan's test.py
        images = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(images)

        with torch.no_grad():
            # initial line drawing
            ld_images = generator(images)

            if deep_sampling:
                for _ in range(2):
                    print("Deep Sampling", _)
                    ld_images = generator(ld_images / 5 + images)

            ld_images = ld_images.permute(0, 2, 3, 1)

            return ld_images
