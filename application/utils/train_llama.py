"""
LLaMA Fine-tuning Script using LoRA/PEFT.
Small-dataset training for proof-of-concept on MSc hardware.
"""
import os
import json
import torch
from typing import List, Dict
from pathlib import Path


class LLaMATrainer:
    """
    Fine-tune LLaMA on small financial dataset using LoRA.

    Demonstrates proof-of-concept training on limited hardware.
    NOT intended for production-scale training.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        lora_rank: int = 8,
        lora_alpha: int = 32,
        output_dir: str = "models/llama-finetuned",
        local_model_path: str = None
    ):
        """
        Initialize trainer.

        Args:
            model_name: Base LLaMA model (HuggingFace ID or local path)
            lora_rank: LoRA rank (lower = less memory)
            lora_alpha: LoRA alpha parameter
            output_dir: Directory to save fine-tuned model
            local_model_path: Path to locally downloaded model (overrides model_name)
        """
        # Use local path if provided, otherwise use model_name
        if local_model_path and os.path.exists(local_model_path):
            self.model_name = local_model_path
            print(f"Using local model from: {local_model_path}")
        else:
            self.model_name = model_name
            print(f"Using HuggingFace model: {model_name}")

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.output_dir = output_dir

        print(f"Initializing LLaMA Trainer")
        print(f"LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")

    def load_training_data(self, data_path: str) -> List[Dict]:
        """
        Load training data from JSON file.

        Expected format:
        [
            {"question": "...", "answer": "..."},
            ...
        ]

        Args:
            data_path: Path to training data JSON

        Returns:
            List of training examples
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} training examples")
        return data

    def prepare_dataset(self, examples: List[Dict]):
        """
        Prepare dataset for training.

        Args:
            examples: List of Q&A pairs

        Returns:
            Prepared dataset
        """
        try:
            from datasets import Dataset

            # Format as instruction-following dataset
            formatted_data = []
            for example in examples:
                formatted_data.append({
                    'text': f"### Question: {example['question']}\n### Answer: {example['answer']}"
                })

            dataset = Dataset.from_list(formatted_data)
            print(f"Prepared dataset with {len(dataset)} examples")

            return dataset

        except ImportError:
            print("Error: 'datasets' library not installed")
            return None

    def setup_model_and_tokenizer(self):
        """
        Load base model and tokenizer with LoRA configuration.

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType

            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=os.path.exists(self.model_name)  # Use local if path exists
            )
            tokenizer.pad_token = tokenizer.eos_token

            print("Loading base model...")
            print(f"  Path: {self.model_name}")
            print(f"  Using 8-bit quantization for memory efficiency")

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,  # Use 8-bit quantization for memory efficiency
                device_map='auto',
                torch_dtype=torch.float16,
                local_files_only=os.path.exists(self.model_name)  # Use local if path exists
            )

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["q_proj", "v_proj"],  # Target attention layers
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            print("Applying LoRA configuration...")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            return model, tokenizer

        except ImportError as e:
            print(f"Error: Required libraries not installed: {e}")
            return None, None

    def train(
        self,
        train_data: List[Dict],
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """
        Train model on dataset.

        Args:
            train_data: Training examples
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Training metrics
        """
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

            print("\nSetting up training...")

            model, tokenizer = self.setup_model_and_tokenizer()
            if model is None or tokenizer is None:
                return None

            dataset = self.prepare_dataset(train_data)
            if dataset is None:
                return None

            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, max_length=512)

            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=100,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                fp16=True,  # Use mixed precision
                gradient_accumulation_steps=4,  # Simulate larger batch size
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator
            )

            print("\nStarting training...")
            print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
            print("Note: This is a proof-of-concept on limited dataset and hardware\n")

            # Train
            train_result = trainer.train()

            # Save model
            print(f"\nSaving model to {self.output_dir}")
            trainer.save_model()
            tokenizer.save_pretrained(self.output_dir)

            return train_result.metrics

        except Exception as e:
            print(f"Training error: {e}")
            return None

    def evaluate(self, test_data: List[Dict]):
        """
        Evaluate model on test dataset.

        Args:
            test_data: Test examples

        Returns:
            Evaluation metrics
        """
        print("\nEvaluating model...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
            model = AutoModelForCausalLM.from_pretrained(
                self.output_dir,
                device_map='auto'
            )

            correct = 0
            total_latency = 0

            for i, example in enumerate(test_data[:10]):  # Sample 10 examples
                question = example['question']
                expected = example['answer']

                import time
                start = time.time()

                inputs = tokenizer(question, return_tensors='pt').to(model.device)
                outputs = model.generate(**inputs, max_length=200)
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

                latency = (time.time() - start) * 1000
                total_latency += latency

                print(f"\nExample {i+1}:")
                print(f"Q: {question}")
                print(f"A: {generated}")
                print(f"Latency: {latency:.0f}ms")

            avg_latency = total_latency / len(test_data[:10])

            return {
                'avg_latency_ms': avg_latency,
                'examples_tested': min(10, len(test_data))
            }

        except Exception as e:
            print(f"Evaluation error: {e}")
            return None


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune LLaMA on financial Q&A')
    parser.add_argument('--data', required=True, help='Path to training data JSON')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--output', default='models/llama-finetuned', help='Output directory')

    args = parser.parse_args()

    # Initialize trainer
    trainer = LLaMATrainer(output_dir=args.output)

    # Load data
    data = trainer.load_training_data(args.data)

    # Split into train/test
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"\nTrain examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # Train
    metrics = trainer.train(train_data, epochs=args.epochs, batch_size=args.batch_size)

    if metrics:
        print("\nTraining completed!")
        print(f"Metrics: {metrics}")

        # Evaluate
        eval_metrics = trainer.evaluate(test_data)
        if eval_metrics:
            print(f"\nEvaluation: {eval_metrics}")

    print("\n" + "="*60)
    print("PROOF-OF-CONCEPT TRAINING COMPLETE")
    print("="*60)
    print("Limitations:")
    print("- Small dataset (hundreds/thousands of examples)")
    print("- Limited hardware (CPU/single GPU)")
    print("- LoRA fine-tuning only (not full model training)")
    print("- Intended for academic demonstration purposes")
    print("="*60)


def start_training_job(files: List[str], epochs: int = 3, batch_size: int = 4, learning_rate: float = 0.0002) -> str:
    """
    Start a background training job.

    Args:
        files: List of training file paths
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Job ID for tracking
    """
    import uuid
    import threading

    job_id = f"train_{uuid.uuid4().hex[:8]}"

    def run_training():
        trainer = LLaMATrainer()
        all_data = []
        for file_path in files:
            data = trainer.load_training_data(file_path)
            all_data.extend(data)

        trainer.train(all_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return job_id


def get_training_status(job_id: str) -> Dict:
    """
    Get status of training job.

    Args:
        job_id: Training job ID

    Returns:
        Status dictionary
    """
    return {
        'job_id': job_id,
        'status': 'running',
        'progress': 0.5,
        'message': 'Training in progress (mock status)'
    }


if __name__ == '__main__':
    main()
