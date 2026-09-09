"""CLI entry point to execute Phase B Mixed-Precision Experiments."""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from uaqe.optimizer.mixed_precision_experimenter import MixedPrecisionExperimenter

def main():
    experimenter = MixedPrecisionExperimenter()
    report = experimenter.run_all_experiments()
    print("\nPhase B Experiments Finished Successfully!")
    print(f"Outcome Classification: {report.get('decision')}")

if __name__ == "__main__":
    main()
