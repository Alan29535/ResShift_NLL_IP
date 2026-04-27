#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import util_common, util_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the ResShift forward process for a single image pair."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bicx4_swinunet_lpips.yaml"),
        help="Diffusion config used to build the forward schedule.",
    )
    parser.add_argument(
        "--gt-image",
        type=Path,
        default=Path("testdata/Bicubicx4/gt/ILSVRC2012_val_00003445.png"),
        help="Ground-truth image path.",
    )
    parser.add_argument(
        "--lq-image",
        type=Path,
        default=None,
        help="Optional low-quality image path. Defaults to the matching file in lq_matlab.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of forward steps to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for the fixed noise tensor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Torch device for the forward process.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out/forward_process"),
        help="Directory for plots and per-step images.",
    )
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return torch.device(device_name)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    img = util_image.imread(image_path, chn="rgb", dtype="float32")
    tensor = util_image.img2tensor(img).to(device)
    return (tensor - 0.5) / 0.5


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    return util_image.tensor2img(
        tensor.detach().cpu(), rgb2bgr=False, out_type=np.uint8, min_max=(-1, 1)
    )


def bicubic_compatible(tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    if tensor.device.type == "mps":
        return F.interpolate(
            tensor.to("cpu"), scale_factor=scale_factor, mode="bicubic"
        ).to(tensor.device)
    return F.interpolate(tensor, scale_factor=scale_factor, mode="bicubic")


def save_rgb_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)


def plot_grid(images, titles, save_path: Path, ncols: int = 5):
    n_images = len(images)
    nrows = int(np.ceil(n_images / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 3.1))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.flat:
        ax.axis("off")

    for idx, (image, title) in enumerate(zip(images, titles)):
        ax = axes[idx // ncols, idx % ncols]
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    gt_path = args.gt_image
    lq_path = args.lq_image
    if lq_path is None:
        lq_path = gt_path.parent.parent / "lq_matlab" / gt_path.name

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not lq_path.exists():
        raise FileNotFoundError(lq_path)

    with open(args.config, "r", encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)
    configs["diffusion"]["params"]["steps"] = args.steps
    diffusion = util_common.instantiate_from_config(configs["diffusion"])

    x_start = load_image_tensor(gt_path, device)
    y_lq = load_image_tensor(lq_path, device)
    scale_factor = int(configs["diffusion"]["params"].get("sf", 1))
    y_match = bicubic_compatible(y_lq, scale_factor=scale_factor)

    if x_start.shape != y_match.shape:
        raise ValueError(
            f"Shape mismatch after upsampling: gt={tuple(x_start.shape)}, y={tuple(y_match.shape)}"
        )

    noise = torch.randn_like(x_start)
    forward_tensors = []
    forward_titles = []

    for t_idx in range(diffusion.num_timesteps):
        t = torch.tensor([t_idx], device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x_start, y_match, t, noise=noise)
        forward_tensors.append(x_t)
        forward_titles.append(f"t={t_idx + 1}/{diffusion.num_timesteps}")

    stem = gt_path.stem
    image_outdir = args.outdir / stem
    image_outdir.mkdir(parents=True, exist_ok=True)

    gt_rgb = tensor_to_rgb_image(x_start)
    lq_rgb = tensor_to_rgb_image(y_lq)
    lq_up_rgb = tensor_to_rgb_image(y_match)

    save_rgb_image(image_outdir / "gt.png", gt_rgb)
    save_rgb_image(image_outdir / "lq.png", lq_rgb)
    save_rgb_image(image_outdir / "lq_upsampled.png", lq_up_rgb)

    forward_images = []
    for idx, x_t in enumerate(forward_tensors):
        rgb = tensor_to_rgb_image(x_t)
        forward_images.append(rgb)
        save_rgb_image(image_outdir / f"step_{idx + 1:02d}_t{idx:02d}.png", rgb)

    plot_grid(
        forward_images,
        forward_titles,
        image_outdir / "forward_grid.png",
        ncols=5,
    )
    plot_grid(
        [gt_rgb, lq_up_rgb, forward_images[-1]],
        ["GT (x0)", "Upsampled LQ (y)", f"Last step (t={diffusion.num_timesteps})"],
        image_outdir / "reference_triplet.png",
        ncols=3,
    )

    print(f"Config: {args.config}")
    print(f"GT image: {gt_path}")
    print(f"LQ image: {lq_path}")
    print(f"Device: {device}")
    print(f"Forward steps: {diffusion.num_timesteps}")
    print(f"Saved outputs to: {image_outdir}")


if __name__ == "__main__":
    main()
