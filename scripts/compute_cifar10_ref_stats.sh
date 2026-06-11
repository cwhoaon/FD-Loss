#!/usr/bin/env bash
# Compute CIFAR-10 train-set reference statistics for FD/FID.

set -euo pipefail

: "${DATA_ROOT:=data/cifar10}"
: "${OUTPUT_DIR:=data/fid_stats/cifar10}"
: "${GPUS_PER_NODE:=1}"
: "${MASTER_PORT:=29502}"
: "${IMG_SIZE:=32}"
: "${DOWNLOAD_CIFAR10:=0}"

DOWNLOAD_FLAG=()
if [ "$DOWNLOAD_CIFAR10" = "1" ]; then
    DOWNLOAD_FLAG=(--download)
fi

run_stats() {
    local model="$1"
    local output_name="$2"
    local repr_input_size="${3:-256}"

    torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT"         compute_repr_stats.py         --dataset cifar10         --data_path "$DATA_ROOT"         --img_size "$IMG_SIZE"         --model "$model"         --target_size "$repr_input_size"         --output_dir "$OUTPUT_DIR"         --output_name "$output_name"         "${DOWNLOAD_FLAG[@]}"
}

run_stats inception inception_in32_t299_stats.npz 299
run_stats convnext convnext_in32_t224_stats.npz 224
run_stats vit_large_patch14_dinov2.lvd142m vit_large_patch14_dinov2_lvd142m_in32_t256_stats.npz 256
run_stats vit_large_patch14_clip_224.openai vit_large_patch14_clip_224_openai_in32_t256_stats.npz 256
run_stats vit_large_patch16_224.mae vit_large_patch16_224_mae_in32_t224_stats.npz 224
run_stats vit_so400m_patch16_siglip_256.v2_webli vit_so400m_patch16_siglip_256_v2_webli_in32_t224_stats.npz 224
