#!/bin/bash
#SBATCH --job-name=grpo_sweep
#SBATCH --account=bfoz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -e

export MKL_THREADING_LAYER=GNU
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="Qwen/Qwen2.5-3B-Instruct"
CKPT_ROOT="checkpoints"
LAMBDAS=(0.0 0.1 0.3 0.5)

mkdir -p "${CKPT_ROOT}"

# ---- Resume lambda=0.0 from checkpoint-250 if adapter not yet saved ----
LAM=0.0
OUT="${CKPT_ROOT}/grpo_gsm8k_lam${LAM}"
if [ ! -f "${OUT}/adapter_model.safetensors" ]; then
    RESUME=""
    if [ -d "${OUT}/checkpoint-250" ]; then
        RESUME="--resume_from_checkpoint ${OUT}/checkpoint-250"
    fi
    echo ">>> Training lambda=${LAM} -> ${OUT} ${RESUME}"
    python train_grpo.py \
        --model "${MODEL}" \
        --benchmark gsm8k \
        --lambda_len "${LAM}" \
        --output_dir "${OUT}" \
        --num_train_epochs 1 \
        ${RESUME}
else
    echo ">>> Skipping lambda=${LAM} (adapter already exists)"
fi

# ---- Train remaining lambdas ----
for LAM in 0.1 0.3 0.5; do
    OUT="${CKPT_ROOT}/grpo_gsm8k_lam${LAM}"
    if [ -d "${OUT}" ] && [ -f "${OUT}/adapter_model.safetensors" ]; then
        echo ">>> Skipping training for lambda=${LAM} (adapter already at ${OUT})"
        continue
    fi
    echo ">>> Training lambda=${LAM} -> ${OUT}"
    python train_grpo.py \
        --model "${MODEL}" \
        --benchmark gsm8k \
        --lambda_len "${LAM}" \
        --output_dir "${OUT}" \
        --num_train_epochs 1
done

# ---- Evaluate each adapter on all three benchmarks ----
for LAM in "${LAMBDAS[@]}"; do
    ADAPTER="${CKPT_ROOT}/grpo_gsm8k_lam${LAM}"
    for BENCH in gsm8k math500 svamp; do
        MAX_NEW_TOKENS=512
        if [ "${BENCH}" = "math500" ]; then MAX_NEW_TOKENS=1024; fi
        echo ">>> Eval lambda=${LAM} on ${BENCH}"
        python evaluate.py \
            --method grpo \
            --benchmark "${BENCH}" \
            --model "${MODEL}" \
            --adapter "${ADAPTER}" \
            --max_new_tokens "${MAX_NEW_TOKENS}"
    done
done

echo ""
echo "============================================"
echo "  GRPO sweep complete"
echo "  Adapters: ${CKPT_ROOT}/"
echo "  Results:  results/grpo_*/"
echo "============================================"

echo ""
echo "Summary (accuracy vs mean CoT tokens per lambda):"
for f in results/grpo_*/metrics.json; do
    echo ""
    echo "--- ${f} ---"
    cat "${f}"
done
