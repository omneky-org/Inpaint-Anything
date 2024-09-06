from pathlib import Path
from typing import List

import torch
from masks_processing import process_masks
from stable_diffusion_inpaint import fill_img_with_sd
from utils import save_array_to_img


def fill_anything(
    input_img: str = None,
    coords_type: str = "key_in",
    mask_img: str = None,
    point_coords: List[float] = None,
    point_labels: List[int] = [1],
    text_prompt: str = None,
    dilate_kernel_size: int = 50,
    output_dir: str = "./results",
    sam_model_type: str = "vit_h",
    sam_ckpt: str = "sam_vit_h_4b8939.pth",
    seed: int = None,
):
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Masks prediction / Masks processing
    img, masks, out_dir = process_masks(
        input_img=input_img,
        coords_type=coords_type,
        mask_img=mask_img,
        point_coords=point_coords,
        point_labels=point_labels,
        dilate_kernel_size=dilate_kernel_size,
        output_dir=output_dir,
        sam_model_type=sam_model_type,
        sam_ckpt=sam_ckpt,
        device=device,
    )

    # 2. Fill the masked image with Stable Diffusion
    for idx, mask in enumerate(masks):
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Fill the image with the mask
        mask_path = out_dir / f"mask_{idx}.png"
        img_filled_path = out_dir / f"filled_with_{Path(mask_path).name}"
        img_filled = fill_img_with_sd(img, mask, text_prompt, device=device)

        # Save the filled image
        save_array_to_img(img_filled, img_filled_path)
