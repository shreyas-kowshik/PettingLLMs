#!/bin/bash
# Example usage of eval_single_llm.py

# Set your paths here
MODEL_PATH="Qwen/Qwen2.5-Coder-3B"
DATA_PATH="/home/skowshik/PettingLLMs/data/code/test/apps_debug_overfit.parquet"

# Example 1: Basic evaluation with k=1
echo "Running basic evaluation with k=1..."
python eval_single_llm.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --k 5 \
    --max_problems 10 \
    --output_dir outputs/run_coder1

# # Example 2: Evaluation with k=5 for better coverage
# echo "Running evaluation with k=5..."
# python eval_single_llm.py \
#     --model_path "$MODEL_PATH" \
#     --data_path "$DATA_PATH" \
#     --k 5 \
#     --max_problems 10 \
#     --output_dir outputs/run2

# Example 3: Full dataset evaluation
# echo "Running full dataset evaluation..."
# python eval_single_llm.py \
#     --model_path "$MODEL_PATH" \
#     --data_path "$DATA_PATH" \
#     --k 3 \
#     --output_dir outputs/full_eval

echo "Evaluation complete! Check the outputs/ directory for results."

