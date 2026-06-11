import argparse
import logging
import os

import torch
import torchvision
import torch.nn.functional as F
from utils.distributed_util import get_global_rank, get_world_size, is_enabled, is_main_process
from utils.sampling_util import generate_images
from utils.data_util import get_img_save_format
from utils.rng_util import RNGStateManager

logger = logging.getLogger("FD_loss")

_JPEG_MAX_DIM = 65500
_VIS_IMAGES_PER_CLASS = 8


def _save_grid(grid, path):
    """Save a grid image, resizing to half if needed for JPEG dimension limits."""
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        _, h, w = grid.shape
        if h > _JPEG_MAX_DIM or w > _JPEG_MAX_DIM:
            grid = F.interpolate(
                grid.float().unsqueeze(0), size=(h // 2, w // 2), mode="bilinear", antialias=True,
            ).squeeze(0)
    torchvision.utils.save_image(grid, path)

# =============================================================================
# Visualization
# =============================================================================

@torch.inference_mode()
def visualize_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_label: str | None,
    step: int,
    tokenizer: torch.nn.Module | None = None,
    cfg: float = 4.0,
    wandb_logger=None,
):
    """Generate grids for visualisation (no FID computation)."""
    was_training = model.training
    model.eval()
    if getattr(args, "dataset", "imagenet") == "cifar10":
        class_labels = torch.arange(args.num_classes, device="cuda", dtype=torch.long)
    elif args.class_of_interest is not None:
        assert all(0 <= c < args.num_classes for c in args.class_of_interest)
        class_labels = torch.tensor(args.class_of_interest, device="cuda", dtype=torch.long)
    else:
        class_labels = torch.randint(args.num_classes, (8,), device="cuda")
    labels = class_labels.repeat_interleave(_VIS_IMAGES_PER_CLASS)
    total_samples = len(labels)
    world_size = get_world_size()
    rank = get_global_rank()
    chunk_size = (total_samples + world_size - 1) // world_size
    start = min(rank * chunk_size, total_samples)
    end = min(start + chunk_size, total_samples)
    local_labels = labels[start:end]

    same_noise = args.same_noise
    logger.info(
        f"Vis: cfg={cfg}, classes={len(class_labels)}, "
        f"images_per_class={_VIS_IMAGES_PER_CLASS}, total={total_samples}, "
        f"local={len(local_labels)}, ema={ema_label}, same_noise={same_noise}"
    )

    if len(local_labels) > 0:
        local_gen = generate_images(args, model, labels=local_labels, cfg=cfg, tokenizer=tokenizer).cpu()
    else:
        local_gen = torch.empty(0)

    if is_enabled():
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, local_gen)
        if is_main_process():
            gen = torch.cat([x for x in gathered if isinstance(x, torch.Tensor) and x.numel() > 0], dim=0)
        else:
            gen = None
    else:
        gen = local_gen

    if is_main_process():
        assert gen is not None and gen.shape[0] == total_samples, (gen.shape[0], total_samples)
        grid = torchvision.utils.make_grid(gen, _VIS_IMAGES_PER_CLASS, 8, pad_value=1)
        fmt = get_img_save_format(grid)
        path = os.path.join(
            args.vis_dir,
            f"step{step:07d}-cfg={cfg}_ema={ema_label}"
            f"-steps={args.num_sampling_steps}-same_noise={same_noise}.{fmt}",
        )
        torchvision.utils.save_image(grid, path)
        logger.info(f"Saved at {path}")
        if wandb_logger is not None:
            ema_tag = "online" if ema_label is None else str(ema_label)
            noise_tag = "same" if same_noise else "different"
            key = f"visualization/{ema_tag}/steps_{args.num_sampling_steps}/{noise_tag}_noise"
            caption = (
                f"step={step}, cfg={cfg}, ema={ema_tag}, "
                f"sampling_steps={args.num_sampling_steps}, same_noise={same_noise}"
            )
            wandb_logger.log_image(key, path, step=step, caption=caption)

    if is_enabled():
        torch.distributed.barrier()
    torch.cuda.empty_cache()
    if was_training:
        model.train()


# =============================================================================
# Multi-EMA visualization
# =============================================================================

def visualize(args, model, ema_model, step, rng=None, tokenizer=None, wandb_logger=None):
    """Generate visualization grids across all EMA labels, sampling steps, and noise modes."""
    if rng is None:
        rng = RNGStateManager()
        rng.save()

    pre_vis_state = rng.snapshot()
    orig_steps = args.num_sampling_steps

    if len(args.vis_steps) == 0:
        args.vis_steps = [orig_steps]

    for ema_label in list(ema_model.labels) + ["online"]:
        with ema_model.swap(model, label=ema_label):
            for sampling_steps in args.vis_steps:
                args.num_sampling_steps = sampling_steps
                for same_noise in (False, True):
                    args.same_noise = same_noise
                    rng.reset()
                    visualize_generator(
                        args, model, ema_label, step,
                        tokenizer=tokenizer, cfg=args.cfg, wandb_logger=wandb_logger,
                    )

    rng.load(pre_vis_state)
    args.num_sampling_steps = orig_steps
    args.same_noise = False
