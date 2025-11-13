"""
Data Preprocessing Utilities

Cleans and prepares training data for LLaMA fine-tuning.
"""
import os
import json
import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"$%]', '', text)
    text = text.strip()
    return text


def remove_pii(text: str) -> str:
    """
    Remove PII from text using regex patterns.

    Args:
        text: Input text

    Returns:
        Text with PII removed
    """
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    }

    for pattern in patterns.values():
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)

    return text


def tokenize_text(text: str, max_length: int = 512) -> List[str]:
    """
    Tokenize text into chunks.

    Args:
        text: Input text
        max_length: Maximum token length

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def preprocess_file(input_path: str, operations: List[str]) -> Tuple[str, int]:
    """
    Preprocess a training data file.

    Args:
        input_path: Path to input file
        operations: List of operations ('lowercase', 'remove_pii', 'clean', 'tokenize')

    Returns:
        Tuple of (output_path, num_samples)
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply operations
    if 'lowercase' in operations:
        content = content.lower()

    if 'clean' in operations:
        content = clean_text(content)

    if 'remove_pii' in operations:
        content = remove_pii(content)

    # Save preprocessed file
    output_dir = 'training/clean'
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_clean{ext}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Count samples (basic line count)
    num_samples = len(content.split('\n'))

    return output_path, num_samples


def convert_to_alpaca_format(data: List[dict]) -> List[dict]:
    """
    Convert data to Alpaca instruction format.

    Args:
        data: List of Q&A pairs

    Returns:
        List of Alpaca-formatted examples
    """
    alpaca_data = []

    for item in data:
        alpaca_item = {
            'instruction': item.get('question', item.get('query', '')),
            'input': '',
            'output': item.get('answer', item.get('response', ''))
        }
        alpaca_data.append(alpaca_item)

    return alpaca_data


def prepare_training_dataset(
    input_files: List[str],
    output_path: str,
    format: str = 'alpaca'
) -> int:
    """
    Prepare training dataset from multiple files.

    Args:
        input_files: List of input file paths
        output_path: Output file path
        format: Output format ('alpaca', 'jsonl')

    Returns:
        Number of samples processed
    """
    all_data = []

    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    all_data.extend(data.get('examples', []))
            elif file_path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))

    # Convert to target format
    if format == 'alpaca':
        formatted_data = convert_to_alpaca_format(all_data)
    else:
        formatted_data = all_data

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'jsonl':
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')
        else:
            json.dump(formatted_data, f, indent=2)

    return len(formatted_data)
