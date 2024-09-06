from pathlib import Path
from typing import List

import torch
from lama_inpaint import inpaint_img_with_lama
from masks_processing import process_masks
from utils import save_array_to_img


def remove_anything(
    input_img: str = None,
    coords_type: str = "key_in",
    mask_img: str = None,
    point_coords: List[float] = None,
    point_labels: List[int] = [1],
    dilate_kernel_size: int = 15,
    output_dir: str = "./results",
    sam_model_type: str = "vit_h",
    sam_ckpt: str = "sam_vit_h_4b8939.pth",
    lama_config: str = "./lama/configs/prediction/default.yaml",
    lama_ckpt: str = "big-lama",
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

    # 2. Remove (Inpaint the masked images) using LAMA
    for idx, mask in enumerate(masks):
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Inpaint the image with the mask
        mask_path = out_dir / f"mask_{idx}.png"
        img_inpainted_path = out_dir / f"inpainted_with_{Path(mask_path).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, lama_config, lama_ckpt, device=device
        )

        # Save the inpainted image
        save_array_to_img(img_inpainted, img_inpainted_path)
