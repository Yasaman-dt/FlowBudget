#!/usr/bin/env python3
"""Run the shared leading-order EM launcher with 2,000 probes per seed by default."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_sensitivity import main_em


def main(argv=None):
    return main_em(argv, default_counts=[2000])


if __name__ == '__main__':
    main()
