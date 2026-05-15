#!/usr/bin/env bash
# JiT-B FD-loss fine-tuning with auxiliary one-step flow-matching loss.

set -euo pipefail

: "${DATA_ROOT:=./datasets/ImageNet}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${CKPT_PATH:=./checkpoints/trained/checkpoint-80.pth}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${WANDB_SAMPLE_EVERY:=2000}"
: "${FD_FLOW_MATCHING_LOSS_WEIGHT:=0.01}"
: "${FD_FLOW_MATCHING_BATCH_SIZE:=64}"
: "${FD_FLOW_MATCHING_SAMPLE_WEIGHT_MODE:=pred_x0_l2_exp}"
: "${FD_FLOW_MATCHING_SAMPLE_WEIGHT_TEMPERATURE:=10}"


mkdir -p .cache/torchinductor .cache/triton .cache/tmp

export TMPDIR="$PWD/.cache/tmp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"



TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

LOAD_FROM="${CKPT_ROOT}/JiT-B.pth"
if [ -n "$CKPT_PATH" ]; then
    LOAD_FROM="$CKPT_PATH"
fi

FM_ARGS=(
    --fd_flow_matching_loss_weight "$FD_FLOW_MATCHING_LOSS_WEIGHT"
    --fd_flow_matching_sample_weight_mode "$FD_FLOW_MATCHING_SAMPLE_WEIGHT_MODE"
    --fd_flow_matching_sample_weight_temperature "$FD_FLOW_MATCHING_SAMPLE_WEIGHT_TEMPERATURE"
)
if [ -n "$FD_FLOW_MATCHING_BATCH_SIZE" ]; then
    FM_ARGS+=(--fd_flow_matching_batch_size "$FD_FLOW_MATCHING_BATCH_SIZE")
fi

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

run_one() {
    local exp_name="$1"
    shift
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        main_fd.py \
        --project table_2_repurpose_jit_B \
        --exp_name "$exp_name" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "$LOAD_FROM" \
        --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
        --cfg 2.4 --interval_min 0.1 --interval_max 1.0 \
        --ema_type edm \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 50 --online_eval --eval_freq 99 \
        --wandb_sample_every "$WANDB_SAMPLE_EVERY" \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 50 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --auto_resume "${FM_ARGS[@]}" "$WANDB_FLAG" \
        "$@"
}

run_one "JiT-fd-sim-fm${FD_FLOW_MATCHING_LOSS_WEIGHT}-tau${FD_FLOW_MATCHING_SAMPLE_WEIGHT_TEMPERATURE}-ep80" \
    --fd_repr_models "$SIGLIP" "$MAE" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256
