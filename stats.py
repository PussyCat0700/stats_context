import os
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)

def process_hellaswag(ds):
    texts = ds['ctx']
    return [str(text) for text in texts]

def process_arc(ds):
    processed_texts = []
    for item in ds:
        question = item['question']
        choices = item['choices']
        # Combine choices with labels
        choices_text = " ".join([f"{label}: {text}" for text, label in zip(choices['text'], choices['label'])])
        full_text = f"{question} {choices_text}"
        processed_texts.append(full_text)
    return processed_texts

def process_sciqa(ds):
    texts = [item['string'] for item in ds['question']]
    return texts

def calculate_stats(texts):
    token_lengths = [len(tokenizer.encode(text)) for text in texts]
    total_tokens = sum(token_lengths)
    avg_tokens = np.mean(token_lengths)
    num_samples = len(texts)
    return {
        'total_tokens': total_tokens,
        'average_tokens': avg_tokens,
        'num_samples': num_samples
    }

# Process and analyze each dataset
datasets = [
    ("HellaSwag", "Rowan/hellaswag", process_hellaswag),
    ("ARC-Challenge", "allenai/ai2_arc", process_arc, "ARC-Challenge"),
    ("ARC-Easy", "allenai/ai2_arc", process_arc, "ARC-Easy"),
    ("SciQA", "orkg/SciQA", process_sciqa)
]

results = {}

for dataset_info in datasets:
    name = dataset_info[0]
    print(f"Processing {name}...")
    curr_output_path = os.path.join(result_dir, name+'.txt')
    if os.path.exists(curr_output_path):
        continue
    
    # Load dataset
    if len(dataset_info) == 3:
        ds = load_dataset(dataset_info[1])['train']
        processor = dataset_info[2]
    else:
        ds = load_dataset(dataset_info[1], dataset_info[3])['train']
        processor = dataset_info[2]
    
    # Process texts
    texts = processor(ds)
    
    # Calculate statistics
    stats = calculate_stats(texts)
    results[name] = stats
    with open(curr_output_path, 'w') as fw:
        fw.write(f'{stats}\n')

# Print results
print("\nResults:")
for name, stats in results.items():
    print(f"\nDataset: {name}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average tokens per sample: {stats['average_tokens']:.2f}")
    print(f"Number of samples: {stats['num_samples']:,}")