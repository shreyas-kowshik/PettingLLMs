#!/usr/bin/env python3
"""
Minimal standalone script to evaluate Python code against MBPP test cases.
Uses MBPP-specific reward computation (fraction of test cases passed).
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List

# Add parent directory to path to import from pettingllms
sys.path.insert(0, str(Path(__file__).parent.parent))

from pettingllms.multi_agent_env.code.code_utils import compute_mbpp_reward_fraction


async def main():
    """
    Main evaluation function that reads test files and evaluates code.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    txt_dir = script_dir / "txt_mbpp"
    
    # Read the code to test
    code_file = txt_dir / "code.txt"
    ground_truth_file = txt_dir / "ground_truth.txt"  # List of test assertions
    setup_file = txt_dir / "setup.txt"  # Optional setup code
    
    # Check if required files exist
    if not code_file.exists():
        print(f"Error: Required file not found: {code_file}")
        sys.exit(1)
    
    if not ground_truth_file.exists():
        print(f"Error: Required file not found: {ground_truth_file}")
        sys.exit(1)
    
    # Read code
    with open(code_file, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Read ground truth test assertions (one per line, or separated by ---)
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth_content = f.read().strip()
        if "---" in ground_truth_content:
            tests = [t.strip() for t in ground_truth_content.split("---") if t.strip()]
        else:
            tests = [t.strip() for t in ground_truth_content.split("\n") if t.strip()]
    
    # Read setup code (optional)
    setup = ""
    if setup_file.exists():
        with open(setup_file, "r", encoding="utf-8") as f:
            setup = f.read().strip()
    
    if not tests:
        print("Error: No test cases found in ground_truth.txt")
        sys.exit(1)
    
    num_tests = len(tests)
    print(f"Running {num_tests} test(s)...")
    print("=" * 60)
    
    # Evaluate code using MBPP reward computation
    print(f"\nEvaluating code against {num_tests} test assertions...")
    print(f"Code length: {len(code)} characters")
    if setup:
        print(f"Setup code length: {len(setup)} characters")
    
    # Compute fraction of tests passed
    fraction_passed = compute_mbpp_reward_fraction(code, tests, setup)
    
    # Calculate number of passed tests
    passed = int(fraction_passed * num_tests)
    total = num_tests
    
    # Display summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"Success rate: {fraction_passed:.2%}")
    print("=" * 60)
    
    # Show individual test results (we can't get detailed results from compute_mbpp_reward_fraction
    # without modifying it, so we'll just show the summary)
    print(f"\nTest assertions evaluated: {num_tests}")
    print(f"Fraction passed: {fraction_passed:.4f}")
    
    return fraction_passed


if __name__ == "__main__":
    try:
        fraction = asyncio.run(main())
        sys.exit(0 if fraction == 1.0 else 1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

