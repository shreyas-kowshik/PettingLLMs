# Code Evaluation Script

This directory contains a minimal standalone script for evaluating Python code against test cases.

## Usage

1. Place your code in `txt/code.txt`
2. Place test inputs in `txt/test_input.txt` (separate multiple test cases with `---` on a new line)
3. Place expected outputs in `txt/test_output.txt` (separate multiple test cases with `---` on a new line)
4. Run the evaluation:

```bash
python eval_code.py
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

The provided example files test a simple addition program:
- **code.txt**: Reads two numbers and prints their sum
- **test_input.txt**: Three test cases with pairs of numbers
- **test_output.txt**: Expected sums for each test case

Run `python eval_code.py` to see it in action!

