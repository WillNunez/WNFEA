"""
Run the WNFEA Roofline Benchmark & Generate Performance Chart.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wnfea.benchmark.roofline import run_roofline_benchmark

if __name__ == "__main__":
    out_chart = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roofline_benchmark.png")
    run_roofline_benchmark(out_chart)
