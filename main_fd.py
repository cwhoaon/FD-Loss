import argparse
import datetime
import logging
import os
import sys
import time

import torch
import torch.distributed
import torch.nn.functional as F

from utils.builders import create_generation_model, create_tokenizer
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.distributed_util import (
    all_reduce_mean, broadcast_module_params, preempt_requested, register_preempt_handler,
)
from utils.eval_util import evaluate_all_emas, log_wandb_sample_grids
from utils.grad_util import get_grad_norm
from utils.logging_util import MetricLogger, SmoothedValue
from utils.optimizer_util import create_optimizer
from frechet_distance.evaluator import FDEvaluator
from frechet_distance.queue import FeatureQueue
from frechet_distance.losses import (
    compute_frechet_distance_loss,
    diff_all_gather,
    load_mu_and_sigma_reference, precompute_sigma_ref_sqrt,
)
from frechet_distance.repr_models import load_repr_model, model_short_name
from frechet_distance.gan_heads import create_gan_head
from frechet_distance.judges import (
    extract_judge_features, extract_judge_gan_features, resolve_gan_feature_kind,
    resolve_per_model_args, save_fd_queue_states, load_fd_queue_states,
    fill_all_queues, run_sanity_check,
)
from utils.rng_util import RNGStateManager
from utils.schedule_util import adjust_learning_rate
from utils.setup_util import setup
from utils.vis_util import visualize

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# FD train step
# ---------------------------------------------------------------------------

def build_flow_matching_dataloader(args, batch_size=None, log_prefix="[FD+FM]"):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, DistributedSampler

    from utils.data_util import center_crop_arr

    train_dir = os.path.join(args.data_path, "train")
    if not os.path.isdir(train_dir):
        train_dir = args.data_path

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=True,
    ) if args.world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size or args.fd_flow_matching_batch_size or args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    logger.info(
        f"{log_prefix} Using ImageFolder real samples from {train_dir}; "
        f"num_images={len(dataset)}, batch_size={loader.batch_size}"
    )
    return loader, sampler


def infinite_flow_matching_batches(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def compute_one_step_flow_matching_loss(model, x, y, args):
    """One-step t=0 flow-matching loss on random dataset/noise couplings.

    Shapes:
        x:                [B, C, H, W], real images in model space [-1, 1]
        y:                [B], integer class labels
        noise:            [B, C, H, W], sampled independently from N(0, noise_scale^2 I)
        target:           [B, C, H, W], x - noise for one-step noise -> data regression
        pred:             [B, C, H, W], predicted displacement field
        pred_x0:          [B, C, H, W], one-step predicted image f(epsilon, 0)
        per_sample_mse:   [B], mean squared FM regression loss per sample
        sample_weights:   [B], optional multiplicative FM loss weights
    """
    B = x.shape[0]
    device = x.device
    dtype = x.dtype
    noise_scale = getattr(model, "noise_scale", args.noise_scale)
    noise = torch.randn_like(x) * noise_scale
    target = x - noise

    if model.training and hasattr(model, "drop_labels"):
        y = model.drop_labels(y)
    elif model.training and hasattr(model, "label_drop_prob") and hasattr(model, "num_classes"):
        drop = torch.rand(B, device=device) < model.label_drop_prob
        y = torch.where(drop, torch.full_like(y, model.num_classes), y)

    # Loss convention: official flow time s=0 is noise and target dx/ds = x - noise.
    # pMF/iMF samplers store the opposite field u because they update z <- z - h * u.
    if hasattr(model, "u_fn"):
        t = torch.ones(B, device=device)
        h = torch.ones(B, device=device)
        omega = torch.ones(B, device=device)
        t_min = torch.zeros(B, device=device)
        t_max = torch.ones(B, device=device)
        pred = -model.u_fn(noise, t, h, omega, t_min, t_max, y)[0]
    elif hasattr(model, "net") and hasattr(model, "_backbone_t"):
        t = torch.ones(B, device=device)
        x_pred = model.net(noise, model._backbone_t(t), y)
        pred = x_pred - noise
    else:
        raise NotImplementedError(
            f"{type(model).__name__} does not expose a supported t=0 FM prediction path"
        )

    if pred.shape != target.shape:
        raise RuntimeError(
            f"FM prediction shape {tuple(pred.shape)} does not match target {tuple(target.shape)}"
        )

    pred_x0 = noise + pred
    residual = pred - target
    residual_flat = residual.float().reshape(B, -1)
    per_sample_mse = residual_flat.square().mean(dim=1)

    metrics = {
        "fm_t0_unweighted": float(per_sample_mse.mean().detach()),
    }

    if args.fd_flow_matching_sample_weight_mode == "none":
        sample_weights = torch.ones(B, device=device, dtype=per_sample_mse.dtype)
    elif args.fd_flow_matching_sample_weight_mode == "pred_x0_l2_exp":
        pred_error = pred_x0 - x
        pred_error_flat = pred_error.float().reshape(B, -1)
        pred_error_l2 = torch.linalg.vector_norm(pred_error_flat, ord=2, dim=1)
        temperature = args.fd_flow_matching_sample_weight_temperature
        if temperature <= 0.0:
            raise ValueError("--fd_flow_matching_sample_weight_temperature must be > 0")
        log_weights = (-pred_error_l2 / temperature).clamp(min=-80.0, max=0.0)
        sample_weights = torch.exp(log_weights).detach()
        metrics["fm_t0_pred_x0_l2"] = float(pred_error_l2.mean().detach())
        metrics["fm_t0_weight_mean"] = float(sample_weights.mean().detach())
    else:
        raise ValueError(
            f"Unsupported --fd_flow_matching_sample_weight_mode={args.fd_flow_matching_sample_weight_mode}"
        )

    weighted_per_sample_loss = per_sample_mse * sample_weights
    return weighted_per_sample_loss.mean(), metrics


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def gan_heads_state_dict(judges):
    return {
        judge["key"]: judge["gan_head"].state_dict()
        for judge in judges
        if "gan_head" in judge
    }


def load_gan_heads_state_dict(judges, state):
    loaded = 0
    for judge in judges:
        if "gan_head" not in judge:
            continue
        if judge["key"] in state:
            judge["gan_head"].load_state_dict(state[judge["key"]])
            loaded += 1
            logger.info(f"[FD+GAN] Restored discriminator head for '{judge['name']}'")
        else:
            logger.warning(f"[FD+GAN] No discriminator head state for '{judge['name']}'")
    return loaded


def create_gan_optimizer(args, judges):
    params = []
    for judge in judges:
        head = judge.get("gan_head")
        if head is not None:
            params.extend(p for p in head.parameters() if p.requires_grad)
    if not params:
        return None
    opt = torch.optim.AdamW(
        params,
        lr=args.fd_gan_disc_lr,
        betas=(args.fd_gan_beta1, args.fd_gan_beta2),
        weight_decay=args.fd_gan_weight_decay,
    )
    logger.info(f"[FD+GAN] discriminator optimizer = {opt}")
    return opt


def sync_gan_heads(judges, src=0):
    for judge in judges:
        head = judge.get("gan_head")
        if head is not None:
            broadcast_module_params(head, src=src)


def compute_gan_losses(judges, fake_images, real_images, args, gan_optimizer=None):
    """Run optional D update and return generator GAN loss.

    Shapes:
        fake_images: [B, 3, H, W] in [0, 1], gradient path to generator.
        real_images: [B, 3, H, W] in [0, 1].
        discriminator logits: [B].
    """
    if args.fd_gan_loss_weight <= 0.0:
        return fake_images.new_tensor(0.0), {}

    gan_judges = [j for j in judges if "gan_head" in j]
    if not gan_judges:
        return fake_images.new_tensor(0.0), {}

    current_step = getattr(args, "current_step", 0)
    disc_active = current_step >= args.fd_gan_disc_start_step
    gen_active = current_step >= args.fd_gan_gen_start_step
    weight_sum = sum(float(j["weight"]) for j in gan_judges)
    if weight_sum <= 0.0:
        weight_sum = 1.0

    metrics = {
        "gan_d_active": float(disc_active),
        "gan_g_active": float(gen_active),
    }

    if gan_optimizer is not None and disc_active:
        d_metric_accum = {}
        for _ in range(args.fd_gan_d_updates):
            gan_optimizer.zero_grad(set_to_none=True)
            d_total = fake_images.new_tensor(0.0)
            step_metrics = {}
            for judge in gan_judges:
                head = judge["gan_head"]
                metric_name = judge.get("metric_name", judge["name"])
                set_requires_grad(head, True)
                head.train()
                with torch.no_grad():
                    feat_real = extract_judge_gan_features(
                        judge, real_images, judge["gan_feature_kind"],
                    )
                    feat_fake = extract_judge_gan_features(
                        judge, fake_images.detach(), judge["gan_feature_kind"],
                    )
                real_logits = head(feat_real)
                fake_logits = head(feat_fake)
                if real_logits.ndim != 1 or fake_logits.ndim != 1:
                    raise RuntimeError(
                        f"GAN logits must be [B], got real={tuple(real_logits.shape)} "
                        f"fake={tuple(fake_logits.shape)}"
                    )
                d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
                d_total = d_total + (float(judge["weight"]) / weight_sum) * d_loss
                step_metrics[f"gan_d_loss_{metric_name}"] = float(d_loss.detach())
                step_metrics[f"gan_real_logit_{metric_name}"] = float(real_logits.mean().detach())
                step_metrics[f"gan_fake_logit_{metric_name}"] = float(fake_logits.mean().detach())
                step_metrics[f"gan_r1_{metric_name}"] = 0.0

            d_total.backward()
            if torch.distributed.is_initialized():
                for judge in gan_judges:
                    for p in judge["gan_head"].parameters():
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            gan_optimizer.step()
            step_metrics["gan_d_loss"] = float(d_total.detach())
            for key, value in step_metrics.items():
                d_metric_accum[key] = d_metric_accum.get(key, 0.0) + value
        for key, value in d_metric_accum.items():
            metrics[key] = value / args.fd_gan_d_updates

    if not gen_active:
        return fake_images.new_tensor(0.0), metrics

    g_total = fake_images.new_tensor(0.0)
    for judge in gan_judges:
        head = judge["gan_head"]
        metric_name = judge.get("metric_name", judge["name"])
        was_training = head.training
        set_requires_grad(head, False)
        head.eval()
        feat_fake = extract_judge_gan_features(judge, fake_images, judge["gan_feature_kind"])
        fake_logits = head(feat_fake)
        if was_training:
            head.train()
        if fake_logits.ndim != 1:
            raise RuntimeError(f"GAN generator logits must be [B], got {tuple(fake_logits.shape)}")
        g_loss = -fake_logits.mean()
        g_total = g_total + (float(judge["weight"]) / weight_sum) * g_loss
        metrics[f"gan_g_loss_{metric_name}"] = float(g_loss.detach())
        metrics[f"gan_g_loss_norm_{metric_name}"] = float(g_loss.detach())
    for judge in gan_judges:
        set_requires_grad(judge["gan_head"], True)

    metrics["gan_g_loss"] = float(g_total.detach())
    return args.fd_gan_loss_weight * g_total, metrics


def get_fd_train_step(model_wo_ddp, judges, sampling_args, args, tokenizer=None,
                      real_image_iter=None, gan_optimizer=None):
    fid_norm_eps = args.fd_fid_norm_eps
    batch_size = args.batch_size
    num_classes = args.num_classes
    input_shape = (args.input_channels, args.input_size, args.input_size)

    def fd_train_step():
        z = torch.randn(batch_size, *input_shape, device="cuda") * args.noise_scale
        y = torch.randint(0, num_classes, (batch_size,), device="cuda")
        sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)

        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = sampled * 0.5 + 0.5  # [-1,1] -> [0,1]

        loss = torch.tensor(0.0, device="cuda")
        loss_dict = {}
        x_real_01 = None
        y_real = None
        if real_image_iter is not None:
            x_real_u8, y_real = next(real_image_iter)
            x_real_01 = x_real_u8.cuda(non_blocking=True).to(torch.float32).div_(255.0)
            y_real = y_real.cuda(non_blocking=True)

        all_new_feats = []
        for judge in judges:
            feats = extract_judge_features(judge, sampled)
            new_feats = diff_all_gather(feats)
            all_new_feats.append(new_feats)

        for i, judge in enumerate(judges):
            new_feats = all_new_feats[i]
            metric_name = judge.get("metric_name", judge["name"])

            _ns_kwargs = dict(sigma_ref_sqrt=judge.get("sigma_ref_sqrt"))
            if judge["queue"].online_accum or judge["queue"].ema_stats:
                mu, sigma = judge["queue"].build_feats_stats(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    mu=mu, sigma=sigma,
                                                    **_ns_kwargs)
            else:
                all_feats = judge["queue"].build_feats_snapshot(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    all_feats=all_feats,
                                                    **_ns_kwargs)
            fid_loss = fid / (fid.detach() + fid_norm_eps)
            loss = loss + judge["weight"] * fid_loss
            loss_dict[f"fid_{metric_name}"] = float(fid.detach())

        if args.fd_gan_loss_weight > 0.0:
            gan_loss, gan_metrics = compute_gan_losses(
                judges, sampled, x_real_01, args, gan_optimizer=gan_optimizer,
            )
            loss = loss + gan_loss
            loss_dict.update(gan_metrics)

        if args.fd_flow_matching_loss_weight > 0.0:
            x_real_fm = x_real_01.mul(2.0).sub(1.0)
            fm_loss, fm_metrics = compute_one_step_flow_matching_loss(model_wo_ddp, x_real_fm, y_real, args)
            loss = loss + args.fd_flow_matching_loss_weight * fm_loss
            loss_dict["fm_t0"] = float(fm_loss.detach())
            loss_dict.update(fm_metrics)

        loss.backward(create_graph=False)

        if torch.distributed.is_initialized():
            for p in model_wo_ddp.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        for i, judge in enumerate(judges):
            judge["queue"].enqueue(all_new_feats[i].detach())

        return loss, loss_dict

    if args.compile and real_image_iter is not None:
        logger.warning("[Compilation] --compile is disabled because the train step reads a DataLoader")
    elif args.compile:
        from utils.runtime_util import _warmup
        logger.info("[Compilation] Compiling fd_train_step ...")
        t0 = time.perf_counter()
        fd_train_step = torch.compile(fd_train_step)
        _warmup(lambda: fd_train_step(), n=2)
        logger.info(f"[Compilation] fd_train_step compiled in {time.perf_counter() - t0:.2f}s")

    return fd_train_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(args):
    wandb_logger = setup(args)
    register_preempt_handler()

    # -- models, optimizer, checkpoint --
    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    optimizer = create_optimizer(args, model, print_trainable_params=True)
    model_wo_ddp = model

    extra = ckpt_resume(args, model_wo_ddp, optimizer, ema_model,
                        extra_keys=["fd_queue_states", "gan_head_states", "gan_optimizer"])

    rng = RNGStateManager()
    rng.save()
    if (not args.disable_vis) or args.vis_only:
        visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
        if args.vis_only:
            return 0

    # -- frechet distance evaluator --
    repr_model_eval, feat_dim_eval, _, _ = load_repr_model("inception")
    fid_evaluator = FDEvaluator(repr_model_eval, feat_dim_eval, args.fid_stats_path)

    # -- frechet distance system: repr models, queues --
    resolve_per_model_args(args)

    judges = []
    for idx, (name, stats_path, weight, pool_type, ts) in enumerate(zip(
        args.fd_repr_models, args.fd_repr_stats_paths,
        args.fd_repr_weights, args.fd_repr_pool_types, args.fd_target_sizes,
    )):
        repr_model, feat_dim, _, _ = load_repr_model(name, target_size=ts)
        mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type=pool_type)
        queue = FeatureQueue(size=args.queue_size, feat_dim=feat_dim,
                             online_accum=args.fd_online_accum,
                             ema_beta=args.fd_ema_beta).cuda()
        short = model_short_name(name)
        sigma_ref_sqrt = None
        if args.fd_eigvalsh:
            sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref)
        judges.append({
            "key": f"{idx}_{name.replace('.', '_')}",
            "name": short, "metric_name": f"{idx}_{short}", "model": repr_model,
            "feat_dim": feat_dim,
            "pool_type": pool_type,
            "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt,
            "queue": queue, "weight": weight,
        })
        judge = judges[-1]
        if args.fd_gan_loss_weight > 0.0:
            feature_kind = resolve_gan_feature_kind(judge, args.fd_gan_head_type)
            num_prefix_tokens = getattr(repr_model, "num_prefix_tokens", 0)
            judge["gan_feature_kind"] = feature_kind
            judge["gan_head"] = create_gan_head(
                feature_kind=feature_kind,
                c_in=feat_dim,
                c_mid=args.fd_gan_hidden_dim,
                num_prefix_tokens=num_prefix_tokens,
            ).cuda()
            logger.info(
                f"[FD+GAN] Repr '{judge['metric_name']}': head_type={args.fd_gan_head_type}, "
                f"feature_kind={feature_kind}, c_in={feat_dim}, c_mid={args.fd_gan_hidden_dim}, "
                f"num_prefix_tokens={num_prefix_tokens}"
            )
        eig_mode = "eigvalsh" if args.fd_eigvalsh else "eigvals"
        stats_mode = f"ema(beta={args.fd_ema_beta})" if args.fd_ema_beta > 0 else ("online_accum" if args.fd_online_accum else "snapshot")
        logger.info(f"[FD] Repr '{judge['metric_name']}' ({name}): feat_dim={feat_dim}, "
                     f"weight={weight}, pool={pool_type}, stats={stats_path}, "
                     f"eig_mode={eig_mode}, stats_mode={stats_mode}")

    fd_restored = (extra is not None
                   and "fd_queue_states" in extra
                   and load_fd_queue_states(judges, extra["fd_queue_states"]))
    if fd_restored:
        logger.info("[FD] Restored all queue states from checkpoint — skipping queue fill")
        run_sanity_check(judges, args.queue_size, args=args)
    else:
        logger.info(f"[FD] Filling {len(judges)} feature queue(s) "
                    f"({args.queue_size} entries each) ...")
        fill_all_queues(judges, model_wo_ddp, args, tokenizer=tokenizer)
        run_sanity_check(judges, args.queue_size, args=args)
    if args.fd_gan_loss_weight > 0.0:
        sync_gan_heads(judges, src=0)
        logger.info("[FD+GAN] Synchronized discriminator head parameters and buffers from rank 0")
    gan_optimizer = create_gan_optimizer(args, judges) if args.fd_gan_loss_weight > 0.0 else None
    if extra is not None and args.fd_gan_loss_weight > 0.0:
        if "gan_head_states" in extra:
            load_gan_heads_state_dict(judges, extra["gan_head_states"])
        if "gan_optimizer" in extra and gan_optimizer is not None:
            gan_optimizer.load_state_dict(extra["gan_optimizer"])
            logger.info("[FD+GAN] Restored discriminator optimizer")
    del extra
    torch.distributed.barrier()

    model.train()
    args.input_channels = model_wo_ddp.in_channels
    args.input_size = model_wo_ddp.input_size

    real_image_iter = None
    needs_real_images = args.fd_flow_matching_loss_weight > 0.0 or args.fd_gan_loss_weight > 0.0
    if needs_real_images:
        if args.fd_gan_loss_weight > 0.0 and args.fd_gan_d_updates < 1:
            raise ValueError("--fd_gan_d_updates must be >= 1 when GAN is enabled")
        if args.fd_gan_loss_weight > 0.0 and args.fd_gan_disc_start_step < 0:
            raise ValueError("--fd_gan_disc_start_step must be >= 0")
        if args.fd_gan_loss_weight > 0.0 and args.fd_gan_gen_start_step < 0:
            raise ValueError("--fd_gan_gen_start_step must be >= 0")
        if args.fd_gan_loss_weight > 0.0 and args.fd_gan_norm_mode != "none":
            raise NotImplementedError("--gan_norm_mode currently supports only 'none'")
        if args.fd_gan_loss_weight > 0.0 and args.fd_gan_r1_gamma != 0.0:
            raise NotImplementedError("--gan_r1_gamma is exposed for experiments but R1 is not implemented yet")
        if tokenizer is not None:
            if args.fd_flow_matching_loss_weight > 0.0:
                raise NotImplementedError(
                    "--fd_flow_matching_loss_weight currently supports pixel-space models only; "
                    "this tokenizer has no training encode path for dataset samples."
                )
        if (
            args.fd_flow_matching_sample_weight_mode != "none"
            and args.fd_flow_matching_sample_weight_temperature <= 0.0
        ):
            raise ValueError(
                "--fd_flow_matching_sample_weight_temperature must be > 0 "
                f"when mode is {args.fd_flow_matching_sample_weight_mode}"
            )
        real_bsz = args.fd_gan_real_batch_size or args.fd_flow_matching_batch_size or args.batch_size
        real_loader, real_sampler = build_flow_matching_dataloader(
            args, batch_size=real_bsz, log_prefix="[FD+REAL]",
        )
        real_image_iter = infinite_flow_matching_batches(real_loader, real_sampler)

    # -- FD train step closure --
    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }
    fd_train_step = get_fd_train_step(
        model_wo_ddp, judges, sampling_args, args, tokenizer=tokenizer,
        real_image_iter=real_image_iter, gan_optimizer=gan_optimizer,
    )

    # -- training loop --
    logger.info(f"training from step {args.current_step:,} -> {args.total_steps:,} "
                f"({args.start_epoch} -> {args.epochs} epochs)")

    global_bsz = args.batch_size * args.world_size
    ckpt_saver = AsyncCheckpointSaver()
    session_start = time.time()
    step_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # dynamic checkpoint frequency: target ~10 min between saves
    ckpt_target_minutes = 10.0
    ckpt_measure_interval = 1000
    ckpt_timer_start = time.perf_counter()
    ckpt_timer_step = args.current_step
    last_ckpt_step = args.current_step

    # metric logger
    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    for name, window, fmt in [
        ("lr",               1,               "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s",        args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)",  args.print_freq, "{value:.2f}"),
        ("device_mem(GB)",   args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    def _infinite():
        while True:
            yield None

    for step, _ in metric_logger.log_every(
        _infinite(), args.print_freq, header="Train:",
        start_iteration=args.current_step, n_iterations=args.total_steps,
    ):
        model.train()
        adjust_learning_rate(optimizer, step, args)

        loss, loss_dict = fd_train_step()

        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0.0 else get_grad_norm(model.parameters()))

        if torch.isfinite(grad_norm):
            optimizer.step()
            ema_model.step(model)
        else:
            logger.warning(f"[step {step}] NaN/Inf grad_norm — skipping optimizer & EMA update")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += global_bsz

        # timing & metrics
        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()

        loss_value = all_reduce_mean(loss.item())
        loss_dict = {k: all_reduce_mean(v) for k, v in loss_dict.items()}
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        metric_logger.update(
            loss=loss_value, grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{"samples/s/device": sps, "samples/s": sps * args.world_size,
               "samples_seen(M)": args.samples_seen / 1e6, "device_mem(GB)": mem_gb},
            **loss_dict,
        )

        # wandb
        if step % args.print_freq == 0 and wandb_logger:
            elapsed = time.time() - session_start + args.last_elapsed_time
            remaining = args.total_steps - args.current_step
            eta = elapsed / args.current_step * remaining if args.current_step > 0 else 0.0
            elapsed_h = elapsed / 3600
            wandb_logger.update({
                "train/loss": loss_value,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
                "train/samples_seen_M": args.samples_seen / 1e6,
                "perf/samples_per_sec_per_device": sps,
                "perf/samples_per_sec": sps * args.world_size,
                "perf/max_reserved_mem_gb": mem_gb,
                "perf/elapsed_real_hours": elapsed_h,
                "perf/elapsed_device_hours": elapsed_h * args.world_size,
                "perf/eta_real_hours": eta / 3600,
                "perf/eta_device_hours": eta / 3600 * args.world_size,
                **{f"train/{k}": v for k, v in loss_dict.items()},
            }, step=args.current_step)

        # dynamic checkpoint frequency
        steps_since_timer = args.current_step - ckpt_timer_step
        if steps_since_timer >= ckpt_measure_interval:
            elapsed_minutes = (time.perf_counter() - ckpt_timer_start) / 60.0
            minutes_per_step = elapsed_minutes / steps_since_timer
            new_save_every = max(100, round(ckpt_target_minutes / minutes_per_step / 100) * 100)
            if new_save_every != args.save_every:
                logger.info(f"adjusting save_every: {args.save_every} -> {new_save_every} "
                            f"({minutes_per_step * 1000:.1f} min/1k steps)")
                args.save_every = new_save_every
            ckpt_timer_start = time.perf_counter()
            ckpt_timer_step = args.current_step

        # checkpoint
        def _save(saver=ckpt_saver):
            elapsed = time.time() - session_start + args.last_elapsed_time
            fd_extra = {"fd_queue_states": save_fd_queue_states(judges)} if judges else {}
            if args.fd_gan_loss_weight > 0.0:
                fd_extra["gan_head_states"] = gan_heads_state_dict(judges)
                if gan_optimizer is not None:
                    fd_extra["gan_optimizer"] = gan_optimizer.state_dict()
            save_checkpoint(args, step, model_wo_ddp, optimizer, ema_model, elapsed,
                            saver=saver, extra=fd_extra)
            torch.distributed.barrier()

        if (args.current_step - last_ckpt_step >= args.save_every
                or args.current_step == args.total_steps):
            _save()
            last_ckpt_step = args.current_step

        if args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0:
            _save()

        # slurm preemption
        if preempt_requested():
            logger.info(f"Preemption at step {args.current_step}: saving checkpoint ...")
            ckpt_saver.wait()
            _save(saver=None)
            logger.info(f"Preemption checkpoint saved at step {args.current_step}. Exiting.")
            return 0

        # visualization
        if args.vis_every > 0 and args.current_step % args.vis_every == 0:
            visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
            model_wo_ddp.train()

        # fixed 25-class wandb sample grids, independent of metric eval
        if args.wandb_sample_every > 0 and args.current_step % args.wandb_sample_every == 0:
            log_wandb_sample_grids(
                args, model_wo_ddp, ema_model, tokenizer,
                step=args.current_step, wandb_logger=wandb_logger, cfg=args.cfg,
            )
            model_wo_ddp.train()

        # online evaluation
        if args.eval_every > 0 and args.online_eval and args.current_step % args.eval_every == 0:
            torch.cuda.empty_cache()
            evaluate_all_emas(
                args, model_wo_ddp, ema_model, fid_evaluator, tokenizer,
                step=args.current_step, wandb_logger=wandb_logger,
                cfg=args.cfg, num_images=args.num_images_for_eval_and_search,
            )
            model_wo_ddp.train()

    # -- final --
    ckpt_saver.wait()
    total = time.time() - session_start + args.last_elapsed_time
    metric_logger.synchronize_between_processes()
    logger.info(f"averaged stats: {metric_logger}")
    logger.info(f"Training complete. Total time: {datetime.timedelta(seconds=int(total))} "
                f"on {args.world_size} devices")
    torch.cuda.empty_cache()

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("FD loss fine-tuning for generation models", add_help=False)

    # training
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--steps_per_epoch", default=1250, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per GPU")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--same_noise", action="store_true")

    # model architecture
    parser.add_argument("--model", default="pMF_B", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--class_tokens", type=int, default=8)
    parser.add_argument("--time_tokens", type=int, default=4)
    parser.add_argument("--guidance_tokens", type=int, default=4)
    parser.add_argument("--interval_tokens", type=int, default=2)
    parser.add_argument("--norm_eps", type=float, default=0.01)
    parser.add_argument("--norm_p", type=float, default=1.0)
    parser.add_argument("--rope_2d", action="store_true")
    parser.add_argument("--learned_pe", action="store_true")
    parser.add_argument("--disable_v_head", action="store_true")
    parser.add_argument("--t_eps", type=float, default=5e-2)

    # tokenizer
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--token_channels", default=3, type=int)
    parser.add_argument("--tokenizer_patch_size", default=1, type=int)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_sched", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="gradient clip, 0.0 means no clip")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_muon", action="store_true")
    parser.add_argument("--muon_lr", type=float, default=1e-3)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_type", default="edm", type=str, choices=["const", "edm"])
    parser.add_argument("--ema_rates", default=[0.9999, 0.9996], type=float, nargs="+")
    parser.add_argument("--ema_halflife_kimg", default=[250, 500, 1000, 2000], type=float, nargs="+")
    parser.add_argument("--eval_ema_labels", default=None, type=str, nargs="+")

    parser.add_argument("--grad_checkpointing", action="store_true")

    # diffusion / flow-matching
    parser.add_argument("--P_mean", type=float, default=0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--legacy_time_convention", action="store_true")
    parser.add_argument("--tr_uniform", action="store_true")
    parser.add_argument("--ratio_r_neq_t", type=float, default=0.5)
    parser.add_argument("--cfg_beta", type=float, default=1.0)
    parser.add_argument("--cfg_omega_max", type=float, default=7.0)
    parser.add_argument("--aux_head_depth", type=int, default=8)
    parser.add_argument("--loss_type", type=str, default="v", choices=["v", "x"])
    parser.add_argument("--aux_pred_type", type=str, default="v", choices=["v", "x"])
    parser.add_argument("--perceptual_threshold", type=float, default=0.8)
    parser.add_argument("--perceptual_loss_on_aux", action="store_true")

    # sampling & generation
    parser.add_argument("--sampling_method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", default=4.0, type=float)
    parser.add_argument("--cfg_list", type=float, nargs="+",
                        default=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--vis_steps", default=[1], type=int, nargs="+")

    # data
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279],
                        type=int, nargs="+")
    parser.add_argument("--force_class_of_interest", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # checkpointing
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--keep_n_ckpts", default=3, type=int)
    parser.add_argument("--milestone_interval", default=20, type=int)

    # evaluation
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--num_images_for_eval_and_search", default=10000, type=int)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/guided_diffusion_stats.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")

    parser.add_argument("--save_eval_images", action="store_true")
    parser.add_argument("--cfg_min", default=1.0, type=float)
    parser.add_argument("--cfg_max", default=25.0, type=float)
    parser.add_argument("--overwrite_cache", action="store_true")

    # FD fine-tuning
    parser.add_argument("--queue_size", type=int, default=50000)
    parser.add_argument("--fd_fid_norm_eps", type=float, default=0.01)
    parser.add_argument("--fd_queue_fill_bsz", type=int, default=256)
    parser.add_argument("--fd_repr_models", type=str, nargs="+", default=["inception"],
                        help="feature extractors: 'inception' or timm model names")
    parser.add_argument("--fd_repr_stats_paths", type=str, nargs="+", default=None,
                        help="reference stats (.npz) per repr model; auto-inferred if omitted")
    parser.add_argument("--fd_repr_weights", type=float, nargs="+", default=None,
                        help="per-model FID loss weight (default 1.0 each)")
    parser.add_argument("--fd_repr_pool_types", type=str, nargs="+", default=None,
                        help="pool type per repr model: 'cls' or 'avg' (default 'cls')")
    parser.add_argument("--fd_target_sizes", type=int, nargs="+", default=None,
                        help="per-model target resolution override (default: model's native size)")
    parser.add_argument("--fd_online_accum", action="store_true",
                        help="use online accumulators for FD (avoids cloning 50k queue each step)")
    parser.add_argument("--fd_eigvalsh", action="store_true",
                        help="use eigvalsh on symmetric product instead of eigvals (~8x faster, exact)")
    parser.add_argument("--fd_ema_beta", type=float, default=0.0, metavar="BETA",
                        help="EMA decay for FD stats (0=disabled, use queue). "
                             "Implies online_accum. E.g. 0.999 → ~1000-batch window")
    parser.add_argument("--fd_flow_matching_loss_weight", type=float, default=0.0,
                        help="add weighted one-step t=0 flow-matching loss on random real/noise couplings")
    parser.add_argument("--fd_flow_matching_batch_size", type=int, default=None,
                        help="real-image batch size for the auxiliary one-step flow-matching loss "
                             "(default: --batch_size)")
    parser.add_argument("--fd_flow_matching_sample_weight_mode", type=str, default="none",
                        choices=["none", "pred_x0_l2_exp"],
                        help="optional per-sample weighting for the auxiliary FM loss; "
                             "'pred_x0_l2_exp' uses exp(-||f(epsilon,0)-x||_2 / T)")
    parser.add_argument("--fd_flow_matching_sample_weight_temperature", type=float, default=1.0,
                        help="temperature T used by --fd_flow_matching_sample_weight_mode=pred_x0_l2_exp")
    parser.add_argument("--fd_gan_loss_weight", "--gan_loss_weight", dest="fd_gan_loss_weight",
                        type=float, default=0.0,
                        help="global weight for representation-space GAN generator loss")
    parser.add_argument("--fd_gan_disc_lr", "--gan_lr", dest="fd_gan_disc_lr",
                        type=float, default=2e-4,
                        help="discriminator-head AdamW learning rate")
    parser.add_argument("--fd_gan_beta1", "--gan_beta1", dest="fd_gan_beta1",
                        type=float, default=0.0)
    parser.add_argument("--fd_gan_beta2", "--gan_beta2", dest="fd_gan_beta2",
                        type=float, default=0.99)
    parser.add_argument("--fd_gan_weight_decay", "--gan_weight_decay", dest="fd_gan_weight_decay",
                        type=float, default=0.0)
    parser.add_argument("--fd_gan_head_type", "--gan_head_type", dest="fd_gan_head_type",
                        type=str, default="patch",
                        choices=["patch", "scalar"],
                        help="GAN discriminator head: patch uses dense repr features, scalar uses pooled FD features")
    parser.add_argument("--fd_gan_hidden_dim", "--gan_hidden_dim", dest="fd_gan_hidden_dim",
                        type=int, default=512,
                        help="middle channel/hidden width for GAN discriminator heads")
    parser.add_argument("--fd_gan_real_batch_size", "--gan_real_batch_size", dest="fd_gan_real_batch_size",
                        type=int, default=None,
                        help="real-image batch size for GAN D update (default: FM batch size or --batch_size)")
    parser.add_argument("--fd_gan_d_updates", "--gan_d_updates", dest="fd_gan_d_updates",
                        type=int, default=1,
                        help="number of discriminator-head updates per generator update")
    parser.add_argument("--fd_gan_disc_start_step", "--gan_disc_start_step", dest="fd_gan_disc_start_step",
                        type=int, default=0,
                        help="training step at which discriminator-head updates start")
    parser.add_argument("--fd_gan_gen_start_step", "--gan_gen_start_step", dest="fd_gan_gen_start_step",
                        type=int, default=0,
                        help="training step at which generator GAN loss starts contributing to total loss")
    parser.add_argument("--fd_gan_norm_mode", "--gan_norm_mode", dest="fd_gan_norm_mode",
                        type=str, default="none", choices=["none"],
                        help="GAN generator-loss normalization mode; only 'none' is implemented")
    parser.add_argument("--fd_gan_norm_eps", "--gan_norm_eps", dest="fd_gan_norm_eps",
                        type=float, default=0.01,
                        help="reserved GAN norm epsilon for future normalization modes")
    parser.add_argument("--fd_gan_norm_ema_decay", "--gan_norm_ema_decay", dest="fd_gan_norm_ema_decay",
                        type=float, default=0.99,
                        help="reserved GAN norm EMA decay for future normalization modes")
    parser.add_argument("--fd_gan_r1_gamma", "--gan_r1_gamma", dest="fd_gan_r1_gamma",
                        type=float, default=0.0,
                        help="reserved R1 regularization weight; nonzero is not implemented")
    parser.add_argument("--fd_gan_r1_every", "--gan_r1_every", dest="fd_gan_r1_every",
                        type=int, default=16,
                        help="reserved R1 interval for future discriminator regularization")
    # logging & tracking
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--local_eval_dir", type=str, default=None)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--vis_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--wandb_sample_every", type=int, default=1000,
                        help="log fixed 25-class 5x5 sample grids to wandb every N training steps")
    parser.add_argument("--vis_only", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)
    parser.add_argument("--current_step", type=int, default=0)
    parser.add_argument("--samples_seen", type=int, default=0)
    parser.add_argument("--project", default="One3", type=str)
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_false", dest="enable_wandb")

    # system
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dtype", default="bf16", type=str, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sys.exit(train_and_evaluate(args))
