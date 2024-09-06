"""
This will contain a function that processes masks
Either predicts using SAM if coordinates are provided
Else processes the masks directly
"""

from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sam_segment import predict_masks_with_sam
from utils import (
    dilate_mask,
    get_clicked_point,
    load_img_to_array,
    load_mask_to_array,
    save_array_to_img,
    show_mask,
    show_points,
)


def process_masks(
    input_img: str = None,
    coords_type: str = "key_in",
    mask_img: str = None,
    point_coords: List[float] = None,
    point_labels: List[int] = [1],
    dilate_kernel_size: int = 50,
    output_dir: str = "./results",
    sam_model_type: str = "vit_h",
    sam_ckpt: str = "sam_vit_h_4b8939.pth",
    device: str = "cuda",
):
    # Set the latest coordinates if provided
    if coords_type == "click":
        latest_coords = get_clicked_point(input_img)
    elif coords_type == "key_in":
        latest_coords = point_coords
    elif coords_type == "none":
        latest_coords = None

    # Load the image into an array
    img = load_img_to_array(input_img)

    if latest_coords is None and mask_img:
        # if no coordinates provided and if mask image is provided,
        # use the mask image directly and skip SAM predictions
        masks = np.asarray([load_mask_to_array(mask_img)])
    else:
        masks, _, _ = predict_masks_with_sam(
            img,
            [latest_coords],
            point_labels,
            model_type=sam_model_type,
            ckpt_p=sam_ckpt,
            device=device,
        )

        # convert masks to binary
        masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(input_img).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / "with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams["figure.dpi"]
        height, width = img.shape[:2]
        plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img)
        plt.axis("off")
        if latest_coords is not None:
            show_points(
                plt.gca(), [latest_coords], point_labels, size=(width * 0.04) ** 2
            )
            plt.savefig(img_points_p, bbox_inches="tight", pad_inches=0)

        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches="tight", pad_inches=0)
        plt.close()

    return img, masks, out_dir
