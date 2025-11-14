"""
Data Preprocessing Utilities

Cleans and prepares training data for LLaMA fine-tuning.
"""
import os
import json
import re
from typing import List, Tuple, Dict


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
        'account_number': r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b',
    }

    for pattern in patterns.values():
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)

    return text


def preprocess_json_qa_file(input_path: str, operations: List[str] = None) -> Tuple[str, int]:
    """
    Preprocess a JSON Q&A training file.

    Expected input format:
    [
        {"question": "...", "answer": "..."},
        ...
    ]

    Args:
        input_path: Path to input JSON file
        operations: List of operations ('clean', 'remove_pii', 'lowercase')

    Returns:
        Tuple of (output_path, num_samples)
    """
    if operations is None:
        operations = ['clean', 'remove_pii']

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON array of Q&A pairs")

    # Process each Q&A pair
    processed_data = []
    for item in data:
        if 'question' not in item or 'answer' not in item:
            continue

        question = item['question']
        answer = item['answer']
        intent = item.get('intent', 'general')

        # Apply operations
        if 'lowercase' in operations:
            question = question.lower()
            answer = answer.lower()

        if 'clean' in operations:
            question = clean_text(question)
            answer = clean_text(answer)

        if 'remove_pii' in operations:
            question = remove_pii(question)
            answer = remove_pii(answer)

        processed_item = {
            'question': question,
            'answer': answer
        }

        # Preserve intent if present
        if 'intent' in item:
            processed_item['intent'] = intent

        processed_data.append(processed_item)

    # Save preprocessed file
    output_dir = 'training/clean'
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_clean{ext}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    return output_path, len(processed_data)


def preprocess_file(input_path: str, operations: List[str] = None) -> Tuple[str, int]:
    """
    Preprocess a training data file (auto-detects format).

    Args:
        input_path: Path to input file
        operations: List of operations to apply

    Returns:
        Tuple of (output_path, num_samples)
    """
    if operations is None:
        operations = ['clean', 'remove_pii']

    # Detect file type
    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.json':
        return preprocess_json_qa_file(input_path, operations)

    elif ext in ['.txt', '.csv']:
        # Simple text preprocessing
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

        # Count samples (lines)
        num_samples = len(content.split('\n'))

        return output_path, num_samples

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def convert_to_alpaca_format(data: List[Dict]) -> List[Dict]:
    """
    Convert Q&A data to Alpaca instruction format.

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
        format: Output format ('alpaca', 'jsonl', 'json')

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'jsonl':
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')
        else:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    return len(formatted_data)
