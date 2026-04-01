#!/bin/bash
set -e

export MKL_THREADING_LAYER=GNU
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="Qwen/Qwen2.5-3B-Instruct"
ADAPTER="hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K"

echo "============================================"
echo "  CoT Compression Baseline Evaluation"
echo "  Model: ${MODEL}"
echo "============================================"
echo ""

# ---- Standard Chain-of-Thought ----
echo ">>> [1/9] Standard CoT on GSM8K"
python evaluate.py --method cot --benchmark gsm8k --model ${MODEL}

echo ">>> [2/9] Standard CoT on MATH-500"
python evaluate.py --method cot --benchmark math500 --model ${MODEL} --max_new_tokens 1024

echo ">>> [3/9] Standard CoT on SVAMP"
python evaluate.py --method cot --benchmark svamp --model ${MODEL}

# ---- Chain of Draft ----
echo ">>> [4/9] Chain of Draft on GSM8K"
python evaluate.py --method cod --benchmark gsm8k --model ${MODEL}

echo ">>> [5/9] Chain of Draft on MATH-500"
python evaluate.py --method cod --benchmark math500 --model ${MODEL} --max_new_tokens 1024

echo ">>> [6/9] Chain of Draft on SVAMP"
python evaluate.py --method cod --benchmark svamp --model ${MODEL}

# ---- TokenSkip (GSM8K only, pre-trained adapter) ----
echo ">>> [7/9] TokenSkip on GSM8K (ratio=0.7)"
python evaluate.py --method tokenskip --benchmark gsm8k --model ${MODEL} \
    --adapter ${ADAPTER} --compression_ratio 0.7

echo ">>> [8/9] TokenSkip on GSM8K (ratio=0.5)"
python evaluate.py --method tokenskip --benchmark gsm8k --model ${MODEL} \
    --adapter ${ADAPTER} --compression_ratio 0.5

echo ">>> [9/9] TokenSkip on GSM8K (ratio=0.3)"
python evaluate.py --method tokenskip --benchmark gsm8k --model ${MODEL} \
    --adapter ${ADAPTER} --compression_ratio 0.3

echo ""
echo "============================================"
echo "  All baselines complete!"
echo "  Results saved in results/"
echo "============================================"

echo ""
echo "Summary of all results:"
echo "------------------------"
for f in results/*/metrics.json; do
    echo ""
    echo "--- ${f} ---"
    cat "${f}"
done
