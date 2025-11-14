#!/bin/bash
# Example usage of eval_single_llm_mbpp.py for MBPP dataset evaluation

# Set your paths here
MODEL_PATH="Qwen/Qwen3-1.7B"  # Change to your model path
DATA_PATH="/home/skowshik/PettingLLMs/data/code/test/mbpp.parquet"  # Change to your MBPP parquet file path

# Example 1: Basic evaluation with k=1
echo "Running basic MBPP evaluation with k=1..."
python eval_single_llm_mbpp.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --k 1 \
    --max_problems 10 \
    --output_dir outputs/mbpp_run1

# Example 2: Evaluation with k=5 for better coverage
# echo "Running MBPP evaluation with k=5..."
# python eval_single_llm_mbpp.py \
#     --model_path "$MODEL_PATH" \
#     --data_path "$DATA_PATH" \
#     --k 5 \
#     --max_problems 10 \
#     --output_dir outputs/mbpp_run2

# Example 3: Full dataset evaluation
# echo "Running full MBPP dataset evaluation..."
# python eval_single_llm_mbpp.py \
#     --model_path "$MODEL_PATH" \
#     --data_path "$DATA_PATH" \
#     --k 3 \
#     --output_dir outputs/mbpp_full_eval

# Example 4: Quick test with small subset
# echo "Running quick test on 5 problems..."
# python eval_single_llm_mbpp.py \
#     --model_path "$MODEL_PATH" \
#     --data_path "$DATA_PATH" \
#     --k 1 \
#     --max_problems 5 \
#     --output_dir outputs/mbpp_quick_test

echo "Evaluation complete! Check the outputs/ directory for results."

