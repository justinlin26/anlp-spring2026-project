#!/bin/bash
# GRPO length-regularized training on MATH (full train split, ~7.5k problems).
# Mirror of run_grpo.sh but trains on MATH instead of GSM8K.
# Adapters land at ${OUTPUT_ROOT}/checkpoints/grpo_math_lam* so they don't collide
# with the GSM8K-trained adapters.

set -e

export MKL_THREADING_LAYER=GNU
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_ROOT="${OUTPUT_ROOT:-.}"
CKPT_ROOT="${OUTPUT_ROOT}/checkpoints"
RESULTS_ROOT="${OUTPUT_ROOT}/results"
LAMBDAS=(0.0 0.1 0.3 0.5)
MAX_STEPS="${MAX_STEPS:-300}"

# MATH problems need longer completions (baseline is ~580 tokens vs GSM8K's ~310).
# Normalizing the length penalty by 1024 also means lambda units are ~half as strong
# per token as on GSM8K — intentional, so the reward range stays [-lambda, 1].
TRAIN_FLAGS=(
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 8
    --num_generations 4
    --vllm_gpu_memory_utilization 0.25
    --max_completion_length 1024
    --max_steps "${MAX_STEPS}"
)

mkdir -p "${CKPT_ROOT}" "${RESULTS_ROOT}"

echo "============================================"
echo "  GRPO length-regularized training on MATH"
echo "  Base model: ${MODEL}"
echo "  Lambda values: ${LAMBDAS[*]}"
echo "  Max steps per lambda: ${MAX_STEPS}"
echo "============================================"

# ---- Train one adapter per lambda ----
for LAM in "${LAMBDAS[@]}"; do
    OUT="${CKPT_ROOT}/grpo_math_lam${LAM}"
    if [ -d "${OUT}" ] && [ -f "${OUT}/adapter_model.safetensors" ]; then
        echo ">>> Skipping training for lambda=${LAM} (adapter already at ${OUT})"
        continue
    fi
    echo ">>> Training lambda=${LAM} -> ${OUT}"
    python train_grpo.py \
        --model "${MODEL}" \
        --benchmark math \
        --lambda_len "${LAM}" \
        --output_dir "${OUT}" \
        "${TRAIN_FLAGS[@]}"
done

# ---- Evaluate each adapter on all three benchmarks ----
for LAM in "${LAMBDAS[@]}"; do
    ADAPTER="${CKPT_ROOT}/grpo_math_lam${LAM}"
    for BENCH in gsm8k math500 svamp; do
        MAX_NEW_TOKENS=512
        if [ "${BENCH}" = "math500" ]; then MAX_NEW_TOKENS=1024; fi
        echo ">>> Eval lambda=${LAM} on ${BENCH}"
        python evaluate.py \
            --method grpo \
            --benchmark "${BENCH}" \
            --model "${MODEL}" \
            --adapter "${ADAPTER}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --output_dir "${RESULTS_ROOT}/grpo_mathtrained_${BENCH}_lam${LAM}"
    done
done

echo ""
echo "============================================"
echo "  GRPO (MATH-trained) sweep complete"
echo "  Adapters: ${CKPT_ROOT}/grpo_math_lam*/"
echo "  Results:  ${RESULTS_ROOT}/grpo_mathtrained_*/"
echo "============================================"

echo ""
echo "Summary (accuracy vs mean CoT tokens per lambda):"
for f in "${RESULTS_ROOT}"/grpo_mathtrained_*/metrics.json; do
    echo ""
    echo "--- ${f} ---"
    cat "${f}"
done
