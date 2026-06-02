#!/usr/bin/env bash
# enwik8 depth sweep: n_layer in {6,8,10,12}, baseline + dblock B=L/2
# baseline: 30k iters each; dblock: B*30k iters each
# Run from repo root. Adjust PYTHON if needed.
set -euo pipefail

PYTHON="python"
BASE_ITERS=30000

declare -A DEPTHS=(
    [6]=3
    [8]=4
    [10]=5
    [12]=6
)

for L in 6 8 10 12; do
    B=${DEPTHS[$L]}
    DBLOCK_ITERS=$(( B * BASE_ITERS ))

    echo "=== baseline L=${L} (${BASE_ITERS} iters) ==="
    $PYTHON train.py config/train_enwik8_baseline.py \
        --n_layer=${L} \
        --out_dir=out-enwik8-baseline-L${L} \
        --wandb_run_name=baseline-L${L} \
        --max_iters=${BASE_ITERS} \
        --lr_decay_iters=${BASE_ITERS}

    echo "=== dblock L=${L} B=${B} (${DBLOCK_ITERS} iters) ==="
    $PYTHON train_dblock.py config/train_enwik8_dblock.py \
        --n_layer=${L} \
        --num_dblocks=${B} \
        --out_dir=out-enwik8-dblock-L${L}-B${B} \
        --wandb_run_name=dblock-L${L}-B${B} \
        --max_iters=${DBLOCK_ITERS} \
        --lr_decay_iters=${DBLOCK_ITERS}
done

echo "=== sweep complete — running plot ==="
$PYTHON plot_enwik8_scaling.py
