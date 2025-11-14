# mbpp_reward.py
import re
import sys
import multiprocessing as mp
import traceback
from typing import Any, Dict, List, Optional

TIMEOUT_SEC = 10.0

def _exec_with_tests(code: str, setup: str, tests: List[str], verbose=False) -> bool:
    """
    Run user code + tests in a restricted namespace inside a forked process.
    Return True iff all tests pass (no AssertionError/Exception).
    """
    def runner(q):
        try:
            # Restrict builtins (comprehensive set for MBPP tests)
            safe_builtins = {
                "abs": abs, "min": min, "max": max, "sum": sum, "range": range,
                "len": len, "enumerate": enumerate, "float": float, "int": int, "str": str,
                "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
                "sorted": sorted, "reversed": reversed, "zip": zip, "map": map, "filter": filter,
                "all": all, "any": any, "pow": pow, "round": round, "divmod": divmod,
                "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
                "isinstance": isinstance, "issubclass": issubclass, "hasattr": hasattr,
                "getattr": getattr, "setattr": setattr, "delattr": delattr,
                "slice": slice, "iter": iter, "next": next,
                "frozenset": frozenset, "bytes": bytes, "bytearray": bytearray,
                "True": True, "False": False, "None": None
            }
            # Execute everything in the same namespace (glb) so functions can access setup variables
            glb = {"__builtins__": safe_builtins}
            if setup:
                exec(setup, glb)
            exec(code, glb)
            for i, t in enumerate(tests):
                try:
                    exec(t, glb)
                except Exception as e:
                    # Send detailed error info back
                    q.put({
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "failed_test_index": i,
                        "failed_test": t
                    })
                    return
            q.put({"success": True})
        except Exception as e:
            q.put({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "stage": "code_execution"
            })

    q = mp.Queue()
    p = mp.Process(target=runner, args=(q,))
    p.start()
    # p.join(TIMEOUT_SEC)
    p.join()
    if p.is_alive():
        p.terminate()
        p.join()
        print("EXEC ERROR: Timeout")
        return False
    try:
        result = q.get_nowait()
        if isinstance(result, dict):
            if result.get("success"):
                return True
            else:
                if verbose:
                    # Print detailed error information
                    print(f"EXEC ERROR: {result.get('error_type', 'Unknown')}: {result.get('error', 'Unknown error')}")
                    if "failed_test_index" in result:
                        print(f"  Failed test #{result['failed_test_index']}: {result['failed_test']}")
                    elif "stage" in result:
                        print(f"  Failed during: {result['stage']}")
                return False
        else:
            # Legacy boolean return (shouldn't happen with new code)
            return bool(result)
    except Exception as e:
        print(f"EXEC ERROR: Failed to get result from queue: {e}")
        return False

def extract_function_name_from_tests(tests: list) -> str:
    """
    Extract the function name that tests are calling.
    Looks for patterns like: assert func_name(...) or func_name(...)
    Returns the first function name found, or empty string if none found.
    """
    import re
    
    for test in tests:
        # Look for function call patterns: func_name(
        # Match identifier followed by opening parenthesis
        match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
        if match:
            func_name = match.group(1)
            # Filter out common keywords that aren't function names
            if func_name not in ['assert', 'if', 'for', 'while', 'def', 'class', 'return', 'print']:
                return func_name
    
    return ""


def extract_function_name_from_code(code: str) -> str:
    """
    Extract the function name from code using pattern: def function_name(
    
    Args:
        code: Python code containing a function definition
    
    Returns:
        Function name or empty string if not found
    """
    import re
    
    # Match: def followed by whitespace, then identifier, then optional whitespace, then (
    # Pattern: def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    if match:
        return match.group(1)
    
    return ""


def fix_function_name(code: str, tests: list) -> str:
    """
    Fix function name mismatch between generated code and test expectations.
    
    Args:
        code: The generated Python code
        tests: List of test assertions
    
    Returns:
        Code with function name replaced to match test expectations
    """
    import re
    
    # Extract expected function name from tests
    expected_name = extract_function_name_from_tests(tests)
    if not expected_name:
        return code  # Can't determine expected name, return as-is
    
    # Extract actual function name from code using def pattern
    actual_name = extract_function_name_from_code(code)
    if not actual_name:
        return code  # No function definition found
    
    # If names match, no need to fix
    if actual_name == expected_name:
        return code
    
    # Replace function name in definition and any recursive calls
    # Use word boundaries to avoid partial replacements
    pattern = r'\b' + re.escape(actual_name) + r'\b'
    fixed_code = re.sub(pattern, expected_name, code)
    
    return fixed_code


def extract_code_from_markdown(text: str) -> str:
    """
    Extract Python code from markdown fences and remove common leading whitespace.
    Handles formats like:
    ```python
    def func():
        pass
    ```
    """
    import re
    import textwrap
    
    # Try to find code between ```python and ```
    pattern = r'```python\s*\n(.*?)\n\s*```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        code = match.group(1)
        # Use textwrap.dedent to remove common leading whitespace from all lines
        code = textwrap.dedent(code).strip()
        return code
    
    # Try without language specifier
    pattern = r'```\s*\n(.*?)\n\s*```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        code = match.group(1)
        # Use textwrap.dedent to remove common leading whitespace from all lines
        code = textwrap.dedent(code).strip()
        return code
    
    # No markdown fences found, return as is (stripped and dedented)
    return textwrap.dedent(text).strip()

# ---- VeRL custom reward entrypoint ----
# VeRL will load this function dynamically when configured via custom_reward_function.path/name.
# The naive reward manager calls: score = compute_score(data_source, solution_str, reward_model, extra_info)
# We'll also accept (solution_str, reward_model, ...) just in case.
# GSM8k format: def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    Returns a float reward in [0,1].
    Expected calling convention (naive manager):
        compute_score(data_source: str, solution_str: str, ground_truth: tests_list, extra_info: dict) -> float
    
    Note: NaiveRewardManager extracts ground_truth from reward_model["ground_truth"] before passing it here.
    We need to get the full reward_model dict to access the setup field.
    """
    VERBOSE = False
    print("\n"*10)
    cleaned_solution = extract_code_from_markdown(solution_str)
    
    if VERBOSE:
        print("IN COMPUTE SCORE")
        print("---SOLUTION STR:---\n", solution_str)
        print("---CLEANED SOLUTION (before name fix):---\n", cleaned_solution)
    
    # Fix function name to match test expectations
    assert isinstance(ground_truth, list), "ground_truth should be a list"
    assert len(ground_truth) > 0, "ground_truth should not be empty"
    tests = ground_truth
    expected_name = extract_function_name_from_tests(tests)
    if VERBOSE:
        print(f"---EXPECTED FUNCTION NAME:--- {expected_name}")
    cleaned_solution = fix_function_name(cleaned_solution, tests)
    if VERBOSE:
        print("---CLEANED SOLUTION (after name fix):---\n", cleaned_solution)
        print("---GROUND TRUTH:---\n", ground_truth)
        print("EXTRA INFO: ", extra_info)
        print("KWAARGS: ", kwargs)
    
    # cleaned_solution = _strip_code_fences(str(cleaned_solution))
    # print("CLEANED SOLUTION AFTER STRIP: ", cleaned_solution)
    
    # Extract setup from extra_info where we'll store it
    # (This is a workaround since NaiveRewardManager doesn't pass the full reward_model)
    setup = ""
    if extra_info and isinstance(extra_info, dict):
        setup = extra_info.get("setup", "")

    if not tests:
        return 0.0

    ok = _exec_with_tests(cleaned_solution, setup, tests, verbose=VERBOSE)
    if VERBOSE:
        print("OK: ", ok)
        print("\n"*10)
    
    return 1.0 if ok else 0.0