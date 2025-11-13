#!/usr/bin/env python3
"""
Minimal standalone script to evaluate Python code against test cases.
Ports core functionality from code_worker.py.
"""

import os
import sys
import asyncio
import subprocess
import tempfile
import shutil
import textwrap
import signal
from pathlib import Path


async def test_if_eq(x, y):
    """
    Test equality of two outputs ignoring whitespace differences.
    """
    if x is None or y is None:
        return False
    return " ".join(str(x).split()) == " ".join(str(y).split())


async def execute_code(script: str, input_val: str, expected_output: str, timeout: float = 40.0):
    """
    Execute Python code with given input and compare against expected output.
    Returns a dict with execution results.
    """
    # Create temporary directory for execution
    tmp_base = os.path.abspath("tmp")
    os.makedirs(tmp_base, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="eval_exec_", dir=tmp_base)
    script_path = os.path.join(tmpdir, "script.py")
    
    def cleanup_tmpdir():
        if os.path.exists(tmpdir):
            try:
                shutil.rmtree(tmpdir, ignore_errors=False)
            except Exception:
                try:
                    subprocess.run(["rm", "-rf", tmpdir], timeout=5, capture_output=True)
                except Exception:
                    pass
    
    stdin_file = None
    stdout_file = None
    stderr_file = None
    printed_output = None
    
    try:
        # Write the script to file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        
        # Create runner that handles input() calls
        runner_path = os.path.join(tmpdir, "runner.py")
        runner_code = textwrap.dedent(
            """
            import sys, io, typing

            def _main():
                input_data = sys.stdin.read()
                input_lines = iter(input_data.splitlines())

                def fake_input(prompt: str = "") -> str:
                    try:
                        return next(input_lines)
                    except StopIteration:
                        raise EOFError("No more input")

                original_stdin = sys.stdin
                sys.stdin = io.StringIO(input_data)

                context = {
                    "__name__": "__main__",
                    "input": fake_input,
                    "List": typing.List,
                    "Tuple": typing.Tuple,
                    "Optional": typing.Optional,
                }

                try:
                    with open("script.py", "r", encoding="utf-8") as sf:
                        code_text = sf.read()
                    try:
                        exec(code_text, context)
                    except SystemExit:
                        pass
                    except Exception as e:
                        print(f"error: {e}")
                finally:
                    sys.stdin = original_stdin

            if __name__ == "__main__":
                _main()
            """
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner_code)
        
        # Setup stdin/stdout/stderr files
        stdin_path = os.path.join(tmpdir, "stdin.txt")
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        stderr_path = os.path.join(tmpdir, "stderr.txt")
        
        with open(stdin_path, "w", encoding="utf-8") as f_in:
            f_in.write(input_val)
        
        stdin_file = open(stdin_path, "rb")
        stdout_file = open(stdout_path, "wb")
        stderr_file = open(stderr_path, "wb")
        
        # Setup environment
        env = dict(os.environ)
        env.update({
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONWARNINGS": "default",
            "PYTHONTRACEMALLOC": "5",
            "PYTHONIOENCODING": "utf-8",
        })
        
        # Execute the code (use relative path since cwd=tmpdir)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "runner.py",
            stdin=stdin_file,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=tmpdir,
            env=env,
            start_new_session=True,
        )
        
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout - 2)
            rc = proc.returncode
        except asyncio.TimeoutError:
            # Kill the process on timeout
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                with open(stderr_path, "ab") as f_err_append:
                    msg = f"[Timeout after {timeout}s; process killed]\n".encode()
                    f_err_append.write(msg)
            except Exception:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
            rc = None
            printed_output = None
        
        # Read output
        if rc is not None:
            try:
                with open(stdout_path, "rb") as f_out:
                    out_bytes = f_out.read()
            except Exception:
                out_bytes = b""
            try:
                with open(stderr_path, "rb") as f_err:
                    err_bytes = f_err.read()
            except Exception:
                err_bytes = b""
            
            if rc == 0:
                printed_output = out_bytes.decode(errors="replace")
            else:
                err_text = (err_bytes or b"").decode(errors="replace").strip()
                out_text = (out_bytes or b"").decode(errors="replace").strip()
                combined = err_text or out_text
                # Show just the last line for cleaner output if it's a traceback
                if "Traceback (most recent call last):" in combined:
                    last_line = combined.strip().splitlines()[-1]
                    combined = last_line
                printed_output = f"error: exit {rc}: {combined}"
    
    except Exception as e:
        printed_output = f"error: {e}"
    
    finally:
        # Close file handles
        for file_handle in [stdin_file, stdout_file, stderr_file]:
            if file_handle is not None:
                try:
                    if not file_handle.closed:
                        file_handle.close()
                except Exception:
                    pass
        
        # Cleanup temporary directory
        cleanup_tmpdir()
    
    # Check if test passed
    if_passed = await test_if_eq(printed_output, str(expected_output)) if printed_output is not None else False
    
    result = {
        "test_input": input_val,
        "code_execution_output": printed_output,
        "test_output": expected_output,
        "passed": if_passed,
    }
    return result


async def main():
    """
    Main evaluation function that reads test files and evaluates code.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    txt_dir = script_dir / "txt"
    
    # Read the code to test
    code_file = txt_dir / "code.txt"
    test_input_file = txt_dir / "test_input.txt"
    test_output_file = txt_dir / "test_output.txt"
    
    # Check if files exist
    for f in [code_file, test_input_file, test_output_file]:
        if not f.exists():
            print(f"Error: Required file not found: {f}")
            sys.exit(1)
    
    # Read code
    with open(code_file, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Read test inputs and outputs
    with open(test_input_file, "r", encoding="utf-8") as f:
        test_inputs = f.read().strip().split("\n---\n")
    
    with open(test_output_file, "r", encoding="utf-8") as f:
        test_outputs = f.read().strip().split("\n---\n")
    
    # Validate test case counts match
    if len(test_inputs) != len(test_outputs):
        print(f"Error: Mismatch in number of test cases. Inputs: {len(test_inputs)}, Outputs: {len(test_outputs)}")
        sys.exit(1)
    
    num_tests = len(test_inputs)
    print(f"Running {num_tests} test(s)...")
    print("=" * 60)
    
    # Run all tests
    results = []
    for i, (test_in, test_out) in enumerate(zip(test_inputs, test_outputs), 1):
        print(f"\nTest {i}/{num_tests}:")
        result = await execute_code(code, test_in.strip(), test_out.strip())
        results.append(result)
        
        print(f"  Input: {test_in.strip()[:50]}{'...' if len(test_in.strip()) > 50 else ''}")
        print(f"  Expected: {test_out.strip()[:50]}{'...' if len(test_out.strip()) > 50 else ''}")
        print(f"  Got: {result['code_execution_output'] if result['code_execution_output'] else 'None'}")
        print(f"  Status: {'✓ PASSED' if result['passed'] else '✗ FAILED'}")
    
    # Calculate and display summary
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    fraction = passed / total if total > 0 else 0.0
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"Success rate: {fraction:.2%}")
    print("=" * 60)
    
    return fraction


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

