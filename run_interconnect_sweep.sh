#!/bin/bash
# Interconnect experiment: sweep sync_interval K ∈ {1, 10, 100, 1000, never}.
# Forces PCIe-only bandwidth (disables NVLink P2P) for a real bandwidth constraint.
#
# For two-node runs, replace torchrun args with:
#   torchrun --nproc_per_node=1 --nnodes=2 \
#       --rdzv_backend=c10d --rdzv_endpoint=<master-host>:29500

set -e

SYNC_INTERVALS=(1 10 100 1000 999999)

for K in "${SYNC_INTERVALS[@]}"; do
    OUT="out-interconnect-sync${K}"
    echo "=== sync_interval=${K} -> ${OUT} ==="
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
        train_dblock_interconnect.py config/train_interconnect.py \
        --sync_interval="${K}" \
        --out_dir="${OUT}"
    echo "=== done: ${OUT} ==="
done

echo "sweep complete"
