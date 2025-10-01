import subprocess
import sys
from itertools import product

# Define the grid of hyperparameters
thresholds = [0.6, 0.7, 0.8]
chunk_sizes = [512, 1024]
similarity_windows = [2, 3, 4]
skip_windows = [0, 1]
min_sentences = [1, 2]

# Path to your evaluation script
SCRIPT = "local_evaluation.py"

# Iterate over all parameter combinations
for thr, cs, sw, sk, ms in product(thresholds, chunk_sizes, similarity_windows, skip_windows, min_sentences):
    print(f"\n=== Running eval with threshold={thr}, chunk_size={cs}, "
          f"similarity_window={sw}, skip_window={sk}, min_sentences_per_chunk={ms} ===")

    # Build command
    cmd = [
        sys.executable, SCRIPT,
        "--threshold", str(thr),
        "--chunk_size", str(cs),
        "--similarity_window", str(sw),
        "--skip_window", str(sk),
        "--min_sentences_per_chunk", str(ms)
    ]

    # Run and stream output to console
    subprocess.run(cmd, text=True, capture_output=True)

