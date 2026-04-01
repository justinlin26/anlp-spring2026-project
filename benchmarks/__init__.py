from benchmarks.gsm8k import load_gsm8k
from benchmarks.math500 import load_math500
from benchmarks.svamp import load_svamp

BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "svamp": load_svamp,
}
