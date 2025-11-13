# Code Evaluation Scripts

This directory contains scripts for evaluating Python code against test cases, including LLM code generation evaluation.

## Scripts

### 1. `eval_code.py` - Single Code Evaluation

Minimal standalone script for evaluating Python code against test cases.

**Usage:**

1. Place your code in `txt/code.txt`
2. Place test inputs in `txt/test_input.txt` (separate multiple test cases with `---` on a new line)
3. Place expected outputs in `txt/test_output.txt` (separate multiple test cases with `---` on a new line)
4. Run the evaluation:

```bash
python eval_code.py
```

### 2. `eval_single_llm.py` - LLM Evaluation

Evaluate a HuggingFace LLM model on APPS-style coding problems with accuracy@k metrics.

**Usage:**

```bash
python eval_single_llm.py \
    --model_path /path/to/huggingface/model \
    --data_path /path/to/apps.parquet \
    --k 5 \
    --max_problems 100 \
    --output_dir outputs/
```

**Arguments:**
- `--model_path`: Path to HuggingFace model directory (not required with `--debug_gt`)
- `--data_path`: Path to APPS parquet file with schema: `[question, test_input, test_output, solution, generated_example]`
  - `test_input` and `test_output` can be:
    - Lists/arrays (preferred for parquet)
    - Strings with `---` separators (legacy format)
- `--k`: Number of solutions to generate per problem (default: 1)
- `--max_problems`: Optional limit on number of problems to evaluate
- `--output_dir`: Directory for output files (default: `outputs/`)
- `--debug_gt`: Debug mode - use ground truth solutions from dataframe (no model loading)

**Debug Mode (`--debug_gt`):**
Use ground truth solutions from the dataframe instead of generating from LLM:
```bash
python eval_single_llm.py \
    --data_path /path/to/apps.parquet \
    --debug_gt \
    --max_problems 10 \
    --output_dir outputs/debug_run
```
This mode is useful for:
- Testing the evaluation pipeline
- Verifying test cases are correct
- Getting baseline accuracy with known-good solutions
- No model loading required!

**Metrics:**
- **Accuracy@1**: Fraction of test cases passed by the first generated solution
- **Accuracy@k**: Fraction of test cases passed by the best of k generated solutions

**Output:**
- `outputs/summary.txt`: Detailed evaluation summary with overall and per-problem results
- `outputs/responses/problem_*.txt`: Individual files for each problem containing:
  - Problem statement
  - Best model response (out of k solutions)
  - Evaluation results
  - All test cases with expected and actual outputs

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## File Format

### test_input.txt
Each test case's input should be on separate lines, with test cases separated by `---`:

```
input_line_1
input_line_2
---
next_test_input_line_1
---
third_test_input
```

### test_output.txt
Each test case's expected output, separated by `---`:

```
expected_output_1
---
expected_output_2
---
expected_output_3
```

### code.txt
The Python code to evaluate. It should read input using `input()` and print results using `print()`.

## Example

The provided example files test a string parsing problem:
- **code.txt**: Parses a bracket and colon pattern
- **test_input.txt**: Test cases with different bracket patterns
- **test_output.txt**: Expected outputs for each test case

Run `python eval_code.py` to see it in action!

