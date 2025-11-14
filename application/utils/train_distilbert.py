"""
Real DistilBERT Fine-tuning Implementation

Production-ready training with progress tracking and model saving.
"""
import os
import json
import torch
from typing import List, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime


class DistilBERTTrainer:
    """
    Fine-tune DistilBERT for intent classification.

    Production-ready implementation with:
    - Real model training
    - Progress callbacks
    - Checkpointing
    - Evaluation metrics
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 10,
        output_dir: str = "models/distilbert-finetuned",
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize trainer.

        Args:
            model_name: Base DistilBERT model
            num_labels: Number of intent classes
            output_dir: Directory to save fine-tuned model
            progress_callback: Function to call with progress updates
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.progress_callback = progress_callback

        os.makedirs(output_dir, exist_ok=True)

        self.log(f"Initializing DistilBERT Trainer")
        self.log(f"Model: {model_name}")
        self.log(f"Output: {output_dir}")

    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)

        if self.progress_callback:
            self.progress_callback({'log': message})

    def load_training_data(self, data_paths: List[str]) -> List[Dict]:
        """
        Load training data from JSON or CSV files.

        Expected JSON format:
        [
            {"question": "...", "answer": "...", "intent": "..."},
            ...
        ]

        Expected CSV format:
        question,answer,intent
        "...", "...", "..."

        Args:
            data_paths: List of paths to training data files

        Returns:
            List of training examples
        """
        all_data = []

        for path in data_paths:
            self.log(f"Loading {path}")
            try:
                file_ext = path.lower().split('.')[-1]

                if file_ext == 'csv':
                    # Load CSV file
                    import csv
                    with open(path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Ensure required fields exist
                            if 'question' in row or 'text' in row:
                                all_data.append({
                                    'question': row.get('question', row.get('text', '')),
                                    'answer': row.get('answer', ''),
                                    'intent': row.get('intent', 'general')
                                })

                elif file_ext in ['json', 'jsonl']:
                    # Load JSON file
                    with open(path, 'r', encoding='utf-8') as f:
                        if file_ext == 'jsonl':
                            # JSON Lines format
                            for line in f:
                                if line.strip():
                                    data = json.loads(line)
                                    all_data.append(data)
                        else:
                            # Regular JSON
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            elif isinstance(data, dict) and 'examples' in data:
                                all_data.extend(data['examples'])

            except Exception as e:
                self.log(f"Error loading {path}: {e}")

        self.log(f"Loaded {len(all_data)} total examples")
        return all_data

    def prepare_dataset(self, examples: List[Dict], intent_labels: Optional[List[str]] = None):
        """
        Prepare dataset for training.

        Args:
            examples: List of training examples
            intent_labels: List of intent label names

        Returns:
            Tuple of (train_dataset, eval_dataset, label_map)
        """
        try:
            from datasets import Dataset
            from sklearn.model_selection import train_test_split

            # Extract unique intents if not provided
            if not intent_labels:
                intent_labels = list(set(ex.get('intent', 'general') for ex in examples))
                intent_labels.sort()

            self.log(f"Intent labels: {intent_labels}")

            # Create label map
            label_map = {label: idx for idx, label in enumerate(intent_labels)}

            # Prepare data
            texts = []
            labels = []

            for ex in examples:
                # Use question as text
                text = ex.get('question', ex.get('text', ''))
                if not text:
                    continue

                # Get intent label
                intent = ex.get('intent', 'general')
                label = label_map.get(intent, 0)

                texts.append(text)
                labels.append(label)

            # Split into train/eval
            train_texts, eval_texts, train_labels, eval_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )

            train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
            eval_dataset = Dataset.from_dict({'text': eval_texts, 'label': eval_labels})

            self.log(f"Train examples: {len(train_dataset)}")
            self.log(f"Eval examples: {len(eval_dataset)}")

            return train_dataset, eval_dataset, label_map

        except ImportError as e:
            self.log(f"Error: Missing library - {e}")
            return None, None, None

    def train(
        self,
        data_paths: List[str],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        intent_labels: Optional[List[str]] = None
    ):
        """
        Train DistilBERT model.

        Args:
            data_paths: Paths to training data files
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            intent_labels: Optional list of intent labels

        Returns:
            Training metrics dictionary
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                TrainerCallback
            )
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            self.log("Loading data...")
            examples = self.load_training_data(data_paths)

            if not examples:
                self.log("Error: No training data loaded")
                return None

            self.log("Preparing dataset...")
            train_dataset, eval_dataset, label_map = self.prepare_dataset(examples, intent_labels)

            if train_dataset is None:
                return None

            # Save label map
            label_map_path = os.path.join(self.output_dir, 'label_map.json')
            with open(label_map_path, 'w') as f:
                json.dump(label_map, f, indent=2)
            self.log(f"Saved label map to {label_map_path}")

            # Load tokenizer and model
            self.log("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.log("Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(label_map)
            )

            # Tokenize datasets
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=128
                )

            self.log("Tokenizing datasets...")
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)

            # Define metrics
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)

                accuracy = accuracy_score(labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted'
                )

                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # Custom callback for progress updates
            class ProgressCallback(TrainerCallback):
                def __init__(self, trainer_instance, total_steps):
                    self.trainer_instance = trainer_instance
                    self.total_steps = total_steps

                def on_step_end(self, args, state, control, **kwargs):
                    if self.trainer_instance.progress_callback:
                        self.trainer_instance.progress_callback({
                            'current_epoch': state.epoch,
                            'current_step': state.global_step,
                            'total_steps': self.total_steps,
                            'loss': state.log_history[-1].get('loss') if state.log_history else None
                        })

                def on_evaluate(self, args, state, control, metrics, **kwargs):
                    if self.trainer_instance.progress_callback and metrics:
                        self.trainer_instance.progress_callback({
                            'eval_metrics': metrics
                        })

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=os.path.join(self.output_dir, 'logs'),
                logging_steps=10,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
                save_total_limit=2,
            )

            # Calculate total steps
            total_steps = (len(train_dataset) // batch_size) * epochs

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                callbacks=[ProgressCallback(self, total_steps)]
            )

            self.log(f"Starting training...")
            self.log(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
            self.log(f"Total steps: {total_steps}")

            # Train
            train_result = trainer.train()

            # Save final model
            self.log("Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(self.output_dir)

            # Final evaluation
            self.log("Running final evaluation...")
            eval_result = trainer.evaluate()

            results = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result.get('eval_loss'),
                'accuracy': eval_result.get('eval_accuracy'),
                'precision': eval_result.get('eval_precision'),
                'recall': eval_result.get('eval_recall'),
                'f1': eval_result.get('eval_f1'),
                'num_train_examples': len(train_dataset),
                'num_eval_examples': len(eval_dataset),
                'output_dir': self.output_dir
            }

            self.log(f"Training completed!")
            self.log(f"Accuracy: {results['accuracy']:.4f}")
            self.log(f"F1 Score: {results['f1']:.4f}")

            return results

        except Exception as e:
            self.log(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return None


def start_distilbert_training(
    data_paths: List[str],
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: str = "models/distilbert-finetuned",
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Start DistilBERT training job.

    Args:
        data_paths: Paths to training data files
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Output directory
        progress_callback: Progress callback function

    Returns:
        Training results
    """
    trainer = DistilBERTTrainer(
        output_dir=output_dir,
        progress_callback=progress_callback
    )

    return trainer.train(
        data_paths=data_paths,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
