#!/usr/bin/env python3
"""
Evaluate a single LLM on MBPP-style coding problems.
Generates k solutions per problem and computes accuracy@1 and accuracy@k.
Uses MBPP-specific reward computation (fraction of test cases passed).
"""

import os
import sys
import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add parent directory to path to import from pettingllms
sys.path.insert(0, str(Path(__file__).parent.parent))

from pettingllms.multi_agent_env.code.code_utils import compute_mbpp_reward_fraction

# Base prompt for MBPP code generation
PROMPT = """You are a helpful Python coding assistant.
Given a task, output ONLY valid Python code that defines the required function(s).
Do not include explanations or markdown fences.
Do not include any docstrings or anything other than the python function(s).
Enclose the entire solution within ```python and ```.

Task:
{prompt}

Constraints:
- Write clean, minimal Python and no other language.
- Define the function(s) exactly as implied by the tests.
- Do NOT print; just return values.
- Output ONLY code (no backticks, no explanations).
- Enclose the entire solution within ```python and ```.
"""


def load_model(model_path: str):
    """Load HuggingFace model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully on device: {model.device}")
    return model, tokenizer


def generate_solutions(model, tokenizer, prompt: str, k: int = 1, max_length: int = 1024) -> List[str]:
    """Generate k code solutions for a given prompt."""
    full_prompt = PROMPT.format(prompt=prompt)
    
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    if torch.cuda.is_available():
        inputs = {key: val.to(model.device) for key, val in inputs.items()}
    
    # Generate k solutions
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_return_sequences=k,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode solutions
    solutions = []
    for output in outputs:
        # Decode and extract code
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Extract code from the generation
        code = extract_code(generated_text, full_prompt)
        solutions.append(code)
    
    return solutions


def extract_code(generated_text: str, prompt: str) -> str:
    """Extract Python code from generated text."""
    # Remove the prompt
    if prompt in generated_text:
        code_part = generated_text[len(prompt):]
    else:
        code_part = generated_text
    
    # Try to extract code from markdown code blocks
    if "```python" in code_part:
        start = code_part.find("```python") + len("```python")
        end = code_part.find("```", start)
        if end != -1:
            code = code_part[start:end].strip()
        else:
            code = code_part[start:].strip()
    elif "```" in code_part:
        start = code_part.find("```") + 3
        end = code_part.find("```", start)
        if end != -1:
            code = code_part[start:end].strip()
        else:
            code = code_part[start:].strip()
    else:
        code = code_part.strip()
    
    return code


async def evaluate_solution(code: str, ground_truth: List[str], setup: str = "", timeout: float = 40.0) -> Dict[str, Any]:
    """Evaluate a code solution against MBPP test cases."""
    if not ground_truth:
        return {"total_tests": 0, "passed_tests": 0, "fraction_passed": 0.0}
    
    # Use MBPP reward computation
    try:
        fraction_passed = compute_mbpp_reward_fraction(code, ground_truth, setup)
        passed = int(fraction_passed * len(ground_truth))
        total = len(ground_truth)
    except Exception as e:
        print(f"Error evaluating solution: {e}")
        fraction_passed = 0.0
        passed = 0
        total = len(ground_truth)
    
    return {
        "total_tests": total,
        "passed_tests": passed,
        "fraction_passed": fraction_passed,
    }


def parse_mbpp_data(row: pd.Series) -> tuple:
    """Parse MBPP data row into prompt, ground_truth, and setup."""
    
    # Get prompt - try different possible column names
    prompt = ''
    if 'prompt' in row:
        prompt = row['prompt']
    elif 'question' in row:
        prompt = row['question']
    
    # Handle NaN/None values
    if pd.isna(prompt):
        prompt = ''
    prompt = str(prompt).strip()
    
    # Extract ground_truth from reward_model
    reward_model = row.get('reward_model', {})
    ground_truth = reward_model.get("ground_truth", [])
    ground_truth = list(ground_truth)
    
    # # Handle different storage formats
    # if pd.isna(reward_model):
    #     reward_model = {}
    # elif isinstance(reward_model, str):
    #     # Try to parse as JSON string
    #     try:
    #         reward_model = json.loads(reward_model)
    #     except (json.JSONDecodeError, TypeError):
    #         reward_model = {}
    
    # if isinstance(reward_model, dict):
    #     ground_truth = reward_model.get('ground_truth', [])
    # elif isinstance(reward_model, list):
    #     ground_truth = reward_model
    # else:
    #     ground_truth = []
    
    # breakpoint()
    
    assert isinstance(ground_truth, list), "ground_truth should be a list, ground_truth: {}, {}".format(type(ground_truth), ground_truth)
    assert len(ground_truth) > 0, "ground_truth should not be empty"
    
    # Extract setup from extra_info
    extra_info = row.get('extra_info', {})
    if isinstance(extra_info, dict):
        setup = extra_info.get('setup', '')
    else:
        setup = ''
    
    # Handle NaN/None in setup
    if pd.isna(setup):
        setup = ''
    setup = str(setup).strip()
    
    return prompt, ground_truth, setup


async def evaluate_single_problem(
    model,
    tokenizer,
    prompt: str,
    ground_truth: List[str],
    setup: str = "",
    k: int = 1,
    use_gt_solution: bool = False,
    gt_solution: str = None
) -> Dict[str, Any]:
    """Evaluate a single problem by generating k solutions or using ground truth."""
    # Use ground truth solution if debug mode is enabled
    if use_gt_solution and gt_solution:
        solutions = [gt_solution]  # Only use GT solution
        print(f"  [DEBUG] Using ground truth solution")
    else:
        # Generate k solutions
        solutions = generate_solutions(model, tokenizer, prompt, k=k)
    
    # Evaluate each solution
    evaluations = []
    for i, code in enumerate(solutions):
        eval_result = await evaluate_solution(code, ground_truth, setup)
        evaluations.append({
            "solution_idx": i,
            "code": code,
            "evaluation": eval_result,
        })
    
    # Compute accuracy@1 (first solution)
    acc_at_1 = evaluations[0]["evaluation"]["fraction_passed"] if evaluations else 0.0
    
    # Compute accuracy@k (best of k solutions)
    acc_at_k = max([e["evaluation"]["fraction_passed"] for e in evaluations]) if evaluations else 0.0
    
    # Find the best solution (highest accuracy)
    best_solution_idx = 0
    if evaluations:
        best_solution_idx = max(range(len(evaluations)), 
                               key=lambda i: evaluations[i]["evaluation"]["fraction_passed"])
    
    return {
        "prompt": prompt,
        "solutions": evaluations,
        "best_solution_idx": best_solution_idx,
        "accuracy_at_1": acc_at_1,
        "accuracy_at_k": acc_at_k,
        "num_solutions": len(evaluations),
        "ground_truth": ground_truth,
        "setup": setup,
    }


def save_problem_response(problem_id: int, result: Dict[str, Any], output_dir: Path, use_gt: bool = False):
    """Save individual problem response to file."""
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(exist_ok=True, parents=True)
    
    response_file = responses_dir / f"problem_{problem_id}.txt"
    
    with open(response_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"PROBLEM {problem_id}\n")
        f.write("=" * 80 + "\n\n")
        
        # Problem statement
        f.write("PROBLEM STATEMENT:\n")
        f.write("-" * 80 + "\n")
        f.write(result["prompt"] + "\n")
        f.write("\n")
        
        # Setup code if present
        if result.get("setup"):
            f.write("SETUP CODE:\n")
            f.write("-" * 80 + "\n")
            f.write(result["setup"] + "\n")
            f.write("\n")
        
        # Best solution
        f.write("=" * 80 + "\n")
        if use_gt:
            f.write("GROUND TRUTH SOLUTION:\n")
        else:
            f.write(f"BEST MODEL RESPONSE (Solution {result['best_solution_idx'] + 1}/{result['num_solutions']}):\n")
        f.write("=" * 80 + "\n")
        best_solution = result["solutions"][result["best_solution_idx"]]
        f.write(best_solution["code"] + "\n")
        f.write("\n")
        
        # Evaluation results for best solution
        f.write("-" * 80 + "\n")
        f.write("BEST SOLUTION EVALUATION:\n")
        f.write("-" * 80 + "\n")
        eval_result = best_solution["evaluation"]
        f.write(f"Tests Passed: {eval_result['passed_tests']}/{eval_result['total_tests']} ")
        f.write(f"({eval_result['fraction_passed']:.2%})\n\n")
        
        # Test cases
        f.write("=" * 80 + "\n")
        f.write("TEST ASSERTIONS:\n")
        f.write("=" * 80 + "\n\n")
        
        for i, test_assertion in enumerate(result["ground_truth"], 1):
            f.write(f"Test Assertion {i}:\n")
            f.write("-" * 40 + "\n")
            f.write(test_assertion + "\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on MBPP coding problems")
    parser.add_argument("--model_path", type=str, required=False, help="Path to HuggingFace model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MBPP parquet file")
    parser.add_argument("--k", type=int, default=1, help="Number of solutions to generate per problem")
    parser.add_argument("--max_problems", type=int, default=None, help="Maximum number of problems to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--debug_gt", action="store_true", help="Use ground truth solutions for debugging (no model loading)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.debug_gt and not args.model_path:
        parser.error("--model_path is required unless --debug_gt is specified")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model (skip if using ground truth)
    if args.debug_gt:
        print("=" * 80)
        print("DEBUG MODE: Using ground truth solutions from dataframe")
        print("=" * 80)
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model(args.model_path)
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} problems")
    
    # Limit number of problems if specified
    if args.max_problems:
        df = df.head(args.max_problems)
        print(f"Evaluating on {len(df)} problems")
    
    # Evaluate each problem
    results = []
    accuracy_at_1_scores = []
    accuracy_at_k_scores = []
    
    print(f"\nEvaluating with k={args.k} solutions per problem...")
    print("=" * 80)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating problems"):
        prompt, ground_truth, setup = parse_mbpp_data(row)
        
        if not prompt or not ground_truth:
            # Debug: print what columns are available
            if idx == 0:
                print(f"\nDebug: Available columns in dataset: {list(df.columns)}")
                print(f"Debug: First row sample - prompt type: {type(row.get('prompt', None))}, reward_model type: {type(row.get('reward_model', None))}")
            print(f"\nSkipping problem {idx}: Missing prompt or test cases (prompt length: {len(prompt)}, ground_truth count: {len(ground_truth)})")
            continue
        
        # Get ground truth solution if in debug mode (MBPP doesn't have solutions, so skip)
        gt_solution = None
        if args.debug_gt:
            print(f"\nSkipping problem {idx}: MBPP dataset doesn't have ground truth solutions")
            continue
        
        try:
            result = await evaluate_single_problem(
                model, tokenizer, prompt, ground_truth, setup,
                k=args.k, use_gt_solution=args.debug_gt, gt_solution=gt_solution
            )
            results.append(result)
            accuracy_at_1_scores.append(result["accuracy_at_1"])
            accuracy_at_k_scores.append(result["accuracy_at_k"])
            
            # Save individual problem response
            save_problem_response(len(results), result, output_dir, use_gt=args.debug_gt)
            
            # Print progress
            if (idx + 1) % 10 == 0:
                avg_acc1 = sum(accuracy_at_1_scores) / len(accuracy_at_1_scores)
                avg_acck = sum(accuracy_at_k_scores) / len(accuracy_at_k_scores)
                print(f"\nProgress: {idx + 1}/{len(df)} | Acc@1: {avg_acc1:.2%} | Acc@{args.k}: {avg_acck:.2%}")
        
        except Exception as e:
            print(f"\nError evaluating problem {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute overall statistics
    if accuracy_at_1_scores:
        overall_acc_at_1 = sum(accuracy_at_1_scores) / len(accuracy_at_1_scores)
        overall_acc_at_k = sum(accuracy_at_k_scores) / len(accuracy_at_k_scores)
    else:
        overall_acc_at_1 = 0.0
        overall_acc_at_k = 0.0
    
    # Write summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LLM Evaluation Summary (MBPP)\n")
        f.write("=" * 80 + "\n\n")
        if args.debug_gt:
            f.write("Mode: DEBUG (Ground Truth Solutions)\n")
        else:
            f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Number of solutions per problem (k): {args.k}\n")
        f.write(f"Problems evaluated: {len(results)}\n\n")
        f.write("-" * 80 + "\n")
        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy@1 (first solution): {overall_acc_at_1:.4f} ({overall_acc_at_1:.2%})\n")
        f.write(f"Accuracy@{args.k} (best of {args.k}): {overall_acc_at_k:.4f} ({overall_acc_at_k:.2%})\n")
        f.write("\n" + "=" * 80 + "\n")
        
        # Detailed results
        f.write("\nDetailed Results per Problem:\n")
        f.write("=" * 80 + "\n")
        for i, result in enumerate(results):
            f.write(f"\nProblem {i + 1}:\n")
            f.write(f"  Prompt: {result['prompt'][:100]}...\n")
            f.write(f"  Accuracy@1: {result['accuracy_at_1']:.2%}\n")
            f.write(f"  Accuracy@{args.k}: {result['accuracy_at_k']:.2%}\n")
            f.write(f"  Solutions generated: {result['num_solutions']}\n")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Problems evaluated: {len(results)}")
    print(f"Accuracy@1: {overall_acc_at_1:.4f} ({overall_acc_at_1:.2%})")
    print(f"Accuracy@{args.k}: {overall_acc_at_k:.4f} ({overall_acc_at_k:.2%})")
    print(f"\nSummary written to: {summary_path}")
    print(f"Individual responses saved to: {output_dir / 'responses'}/")
    print("=" * 80)
    
    return overall_acc_at_1, overall_acc_at_k


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

