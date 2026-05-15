"""Judge system for Frechet Distance training.

Manages repr-model judges: setup, feature extraction, queue filling,
checkpoint save/load, and sanity checks.
"""

import logging

import numpy as np
import torch

from frechet_distance.losses import (
    compute_frechet_distance_loss,
    diff_all_gather,
)
from frechet_distance.metrics import compute_fid as np_fid
from utils.distributed_util import is_main_process

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_stats_path(name, img_size, target_size, default_inception_path=None):
    """Auto-infer reference stats path for a repr model."""
    if name == "inception" and default_inception_path is not None:
        return default_inception_path
    sanitized = name.replace(".", "_")
    if img_size == 512: # TODO
        img_size = 256
    return f"data/fid_stats/{sanitized}_in{img_size}_t{target_size}_stats.npz"


def extract_judge_features(judge, images):
    """Run judge model and return features respecting pool_type.

    TimmReprModel returns (cls_token, mean_token).  When pool_type=='avg',
    we want the mean_token as features.  Inception and CNN models use primary
    features.
    """
    primary, secondary = judge["model"](images)
    if judge.get("pool_type") == "avg":
        return secondary
    return primary


def resolve_gan_feature_kind(judge, head_type):
    """Resolve the feature tensor kind consumed by a GAN discriminator head."""
    if head_type == "scalar":
        return "pooled"
    elif head_type == "patch":
        model = judge["model"]
        if hasattr(model, "forward_spatial"):
            return "cnn_map"
        if hasattr(model, "forward_dense"):
            dense_ndim = getattr(model, "gan_dense_ndim", None)
            if dense_ndim == 4:
                return "cnn_map"
            return "vit_tokens"
        raise ValueError(f"judge {judge['name']} does not expose dense features for patch GAN")
    raise ValueError(f"unsupported GAN head type '{head_type}' for judge {judge['name']}")


def extract_judge_gan_features(judge, images, feature_kind):
    """Extract representation features for GAN heads.

    Shapes by feature_kind:
        pooled:     [B, D]
        vit_tokens: [B, N + P, C]
        cnn_map:    [B, C, H, W]
    """
    if feature_kind == "pooled":
        return extract_judge_features(judge, images)
    model = judge["model"]
    if feature_kind == "cnn_map" and hasattr(model, "forward_spatial"):
        return model.forward_spatial(images)
    if feature_kind in ("vit_tokens", "cnn_map") and hasattr(model, "forward_dense"):
        feats = model.forward_dense(images)
        if feature_kind == "vit_tokens" and feats.ndim != 3:
            raise RuntimeError(f"expected ViT tokens [B, N, C], got {tuple(feats.shape)}")
        if feature_kind == "cnn_map" and feats.ndim != 4:
            raise RuntimeError(f"expected CNN map [B, C, H, W], got {tuple(feats.shape)}")
        return feats
    raise ValueError(f"unsupported GAN feature kind '{feature_kind}' for judge {judge['name']}")


def resolve_per_model_args(args):
    """Resolve per-model stats paths, weights, pool types, and target sizes."""
    num = len(args.fd_repr_models)

    if args.fd_target_sizes is None:
        args.fd_target_sizes = [256] * num
    elif len(args.fd_target_sizes) == 1 and num > 1:
        args.fd_target_sizes *= num

    if args.fd_repr_stats_paths is None:
        args.fd_repr_stats_paths = [
            infer_stats_path(n, args.img_size, ts, args.fid_stats_path)
            for n, ts in zip(args.fd_repr_models, args.fd_target_sizes)
        ]
    elif len(args.fd_repr_stats_paths) == 1 and num > 1:
        args.fd_repr_stats_paths *= num

    if args.fd_repr_weights is None:
        args.fd_repr_weights = [1.0] * num

    if args.fd_repr_pool_types is None:
        args.fd_repr_pool_types = ["cls"] * num
    elif len(args.fd_repr_pool_types) == 1 and num > 1:
        args.fd_repr_pool_types *= num

    assert len(args.fd_target_sizes) == num
    assert len(args.fd_repr_stats_paths) == num
    assert len(args.fd_repr_weights) == num
    assert len(args.fd_repr_pool_types) == num


# ---------------------------------------------------------------------------
# Queue state save/load
# ---------------------------------------------------------------------------

def save_fd_queue_states(judges):
    """Collect queue state dicts from all judges for checkpointing."""
    return [
        {
            "key": j.get("key", j["name"]),
            "name": j["name"],
            "queue": j["queue"].state_dict(),
        }
        for j in judges
    ]


def load_fd_queue_states(judges, saved_states):
    """Restore queue states from checkpoint into judges.

    Matches by stable key when available, with a backward-compatible fallback
    to unique legacy names.
    """
    key_to_state = {
        s["key"]: s["queue"]
        for s in saved_states
        if "key" in s
    }
    name_counts = {}
    for s in saved_states:
        name_counts[s["name"]] = name_counts.get(s["name"], 0) + 1
    unique_name_to_state = {
        s["name"]: s["queue"]
        for s in saved_states
        if name_counts[s["name"]] == 1
    }
    loaded = 0
    for judge in judges:
        state = key_to_state.get(judge.get("key"))
        if state is None:
            state = unique_name_to_state.get(judge["name"])
        if state is not None:
            judge["queue"].load_state_dict(state)
            judge["queue"].cuda()
            loaded += 1
            logger.info(f"[FD] Restored queue state for '{judge.get('metric_name', judge['name'])}'")
        else:
            logger.warning(
                f"[FD] No saved queue state for '{judge.get('metric_name', judge['name'])}', "
                "will need queue fill"
            )
    return loaded == len(judges)


# ---------------------------------------------------------------------------
# Queue filling
# ---------------------------------------------------------------------------

@torch.no_grad()
def fill_all_queues(judges, model, args, tokenizer=None):
    """Fill all repr-model feature queues with generated images.

    EMA judges use streaming accumulation (no feature buffer allocated).
    Non-EMA judges fill the feature buffer as before.
    """
    queue_size = args.queue_size
    if queue_size == 0:
        logger.info("[FD] queue_size=0: skipping queue fill")
        return

    model.eval()
    filled = 0
    while filled < queue_size:
        batch_size = min(args.fd_queue_fill_bsz, queue_size - filled)
        y = torch.randint(0, args.num_classes, (batch_size,), device="cuda")
        imgs = model.generate(batch_size, y, cfg=args.cfg, args=args, verbose=False)
        if tokenizer is not None:
            imgs = tokenizer.detokenize(imgs) # [0, 1]
        else:
            imgs = imgs * 0.5 + 0.5  # [-1,1] -> [0,1]

        for judge in judges:
            local_feats = extract_judge_features(judge, imgs)
            all_feats = diff_all_gather(local_feats)
            count = min(all_feats.shape[0], queue_size - filled)
            q = judge["queue"]
            if q.ema_stats:
                q.accumulate_batch(all_feats[:count])
            else:
                q.feats[filled:filled + count] = all_feats[:count].float()

        filled += count
        logger.info(f"[FD] Queue fill: {filled}/{queue_size} ({filled / queue_size * 100:.1f}%)")

    for judge in judges:
        q = judge["queue"]
        if q.ema_stats:
            q._finalize_streaming_init()
        else:
            q.ptr.zero_()
            if q.online_accum:
                q._init_accumulators()
    logger.info(f"[FD] All {len(judges)} queues initialized with {filled} features")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_sanity_check(judges, queue_size, args=None):
    """Log FID sanity check on rank 0 (numpy vs differentiable).

    Compares numpy FID against the differentiable eigvals baseline.
    When *args* is provided and eigvalsh is enabled,
    also checks those modes and logs the deviation from the baseline.
    """
    if not is_main_process() or queue_size == 0:
        if queue_size == 0:
            logger.info("[FD] queue_size=0: skipping sanity check")
        return

    use_eigvalsh = args.fd_eigvalsh if args else False

    for judge in judges:
        q = judge["queue"]

        if q.ema_stats:
            # EMA mode: no feature buffer — compute FID from EMA moments
            sigma_ema = q.m2_ema - q.mu_ema.unsqueeze(1) * q.mu_ema.unsqueeze(0)
            fid_ema = float(compute_frechet_distance_loss(
                judge["mu_ref"], judge["sigma_ref"],
                mu=q.mu_ema, sigma=sigma_ema,
            ).item())
            parts = [f"FID={fid_ema:.4f} (ema_init)"]
            logger.info(f"[FD] Sanity '{judge['name']}' (feat_dim={judge['feat_dim']}, pool={judge['pool_type']}): {', '.join(parts)}")
            continue

        feats_np = q.feats.cpu().float().numpy()
        fid_numpy = np_fid(
            np.mean(feats_np, 0), np.cov(feats_np, rowvar=False),
            judge["mu_ref"].cpu().numpy(), judge["sigma_ref"].cpu().numpy(),
        )
        fid_diff = compute_frechet_distance_loss(
            judge["mu_ref"], judge["sigma_ref"], all_feats=q.feats,
        ).item()

        parts = [f"FID={fid_numpy:.4f} (numpy)", f"FID={fid_diff:.4f} (diff/eigvals)"]

        sigma_ref_sqrt = judge.get("sigma_ref_sqrt")
        if use_eigvalsh and sigma_ref_sqrt is not None:
            fid_eigvalsh = compute_frechet_distance_loss(
                judge["mu_ref"], judge["sigma_ref"], all_feats=q.feats,
                sigma_ref_sqrt=sigma_ref_sqrt,
            ).item()
            parts.append(f"FID={fid_eigvalsh:.4f} (eigvalsh, err={abs(fid_eigvalsh - fid_diff):.2e})")

        logger.info(f"[FD] Sanity '{judge['name']}' (feat_dim={judge['feat_dim']}, pool={judge['pool_type']}): {', '.join(parts)}")
