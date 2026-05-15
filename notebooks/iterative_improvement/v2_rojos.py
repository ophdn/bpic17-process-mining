"""
v2_rojos.py — Convenience runner: executes all 4 Rojos sub-steps in order
==========================================================================
Runs v2a → v2b → v2c → v2d as separate processes so each step's log is
persisted before the next begins.  You can also run each sub-script
individually for finer control.

Run from any directory:
    python iterative_improvement/v2_rojos.py
"""

import subprocess
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_steps = [
    "v2a_merge_sequences.py",
    "v2b_endpoint_filter.py",
    "v2c_fraud_filter.py",
    "v2d_rare_variants.py",
]

for step in _steps:
    print(f"\n{'='*68}")
    print(f"  Running {step} …")
    print(f"{'='*68}\n")
    result = subprocess.run(
        [sys.executable, os.path.join(_here, step)],
        check=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {step} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)

print("\nAll V2 sub-steps complete.  Next step: python v3_noise_sweep.py")
