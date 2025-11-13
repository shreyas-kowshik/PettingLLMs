from datasets import load_dataset

# Load the dataset (you may need to specify a split, e.g., 'train')
try:
    ds = load_dataset("codeparrot/apps", split="train")
    # Get all unique values in the 'difficulty' column
    difficulty_levels = ds.unique("difficulty")
    print("Valid difficulty levels:", difficulty_levels)

    introductory_examples = ds.filter(lambda example: example["difficulty"] == "introductory")

    # Get the number of rows in the filtered dataset
    count = len(introductory_examples)
    print(f"Number of examples in 'introductory' level: {count}")

except Exception as e:
    # Handle potential errors from previous troubleshooting steps
    print(f"An error occurred: {e}")
