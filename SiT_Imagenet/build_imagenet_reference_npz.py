"""Create a fixed, unguided-FID reference batch from ImageNet training images.

The ADM evaluator used by the official SiT repository expects an ``arr_0`` array
of uint8 images with shape (N, H, W, 3).  This script creates that array with
the same deterministic centre crop used by SiT training (but no random flip).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    arr = np.array(pil_image)
    y = (arr.shape[0] - image_size) // 2
    x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[y : y + image_size, x : x + image_size])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True,
                        help="ImageNet train directory containing the 1,000 class folders.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")
    transform = transforms.Lambda(lambda image: np.asarray(center_crop_arr(image, args.image_size), dtype=np.uint8))
    dataset = ImageFolder(args.data_path, transform=transform)
    if args.num_images > len(dataset):
        raise ValueError(f"Requested {args.num_images} images, but data set has only {len(dataset)}.")
    indices = np.random.default_rng(args.seed).choice(len(dataset), args.num_images, replace=False)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=32, shuffle=False,
                        num_workers=args.workers, pin_memory=False)
    images = np.empty((args.num_images, args.image_size, args.image_size, 3), dtype=np.uint8)
    offset = 0
    for batch, _ in loader:
        batch = np.asarray(batch)
        images[offset : offset + len(batch)] = batch
        offset += len(batch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, arr_0=images)
    print(f"Wrote {args.output} with shape {images.shape}; reference seed={args.seed}.")


if __name__ == "__main__":
    main()
