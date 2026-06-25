"""Compatibility wrapper for HETMCL reliability evaluation."""

from src.evaluation.hetmcl.evaluate_reliability import *  # noqa: F401,F403
from src.evaluation.hetmcl.evaluate_reliability import main


if __name__ == "__main__":
    main()
