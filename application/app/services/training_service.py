"""
Training Service

Manages background training jobs with real progress tracking.
"""
import os
import threading
import uuid
from typing import List, Dict
from datetime import datetime
from app import db
from app.models.training_job import TrainingJob
from app.models.training_file import TrainingFile


class TrainingService:
    """Service for managing model training jobs."""

    def __init__(self):
        self.active_threads = {}

    def start_distilbert_training(
        self,
        training_file_ids: List[int],
        user_id: int,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> str:
        """
        Start DistilBERT training job in background.

        Args:
            training_file_ids: List of training file IDs
            user_id: User ID starting the job
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Job ID
        """
        # Generate job ID
        job_id = f"distilbert_{uuid.uuid4().hex[:8]}"

        # Create job record
        job = TrainingJob(
            job_id=job_id,
            model_type='distilbert',
            status='pending',
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_file_ids=','.join(map(str, training_file_ids)),
            started_by=user_id
        )
        db.session.add(job)
        db.session.commit()

        # Get file paths
        training_files = db.session.query(TrainingFile).filter(
            TrainingFile.id.in_(training_file_ids)
        ).all()

        file_paths = [tf.filepath for tf in training_files]

        # Get current app instance to pass to thread
        from flask import current_app
        app = current_app._get_current_object()

        # Start training in background thread
        thread = threading.Thread(
            target=self._run_distilbert_training,
            args=(app, job.id, file_paths, epochs, batch_size, learning_rate),
            daemon=True
        )
        thread.start()

        self.active_threads[job_id] = thread

        return job_id

    def _run_distilbert_training(
        self,
        app,
        job_id: int,
        file_paths: List[str],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """
        Run DistilBERT training in background.

        Args:
            app: Flask app instance
            job_id: Database job ID
            file_paths: List of training file paths
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        from utils.train_distilbert import start_distilbert_training

        # Create app context for database access
        with app.app_context():
            # Get job
            job = db.session.get(TrainingJob, job_id)
            if not job:
                return

            try:
                # Mark as running
                job.mark_running()
                job.append_log("Training job started")
                job.append_log(f"Training files: {len(file_paths)}")
                job.append_log(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

                # Output directory
                output_dir = f"models/distilbert-{job.job_id}"

                # Progress callback
                def progress_callback(update: Dict):
                    with app.app_context():
                        job_obj = db.session.get(TrainingJob, job_id)
                        if not job_obj:
                            return

                        # Update progress
                        if 'current_epoch' in update:
                            job_obj.update_progress(
                                current_epoch=int(update.get('current_epoch', 0)),
                                current_step=update.get('current_step', 0),
                                total_steps=update.get('total_steps', 1),
                                metrics={'loss': update.get('loss')}
                            )

                        # Update eval metrics
                        if 'eval_metrics' in update:
                            metrics = update['eval_metrics']
                            job_obj.eval_loss = metrics.get('eval_loss')
                            job_obj.accuracy = metrics.get('eval_accuracy')
                            db.session.commit()

                        # Append logs
                        if 'log' in update:
                            job_obj.append_log(update['log'])

                # Start training
                results = start_distilbert_training(
                    data_paths=file_paths,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    output_dir=output_dir,
                    progress_callback=progress_callback
                )

                if results:
                    # Mark as completed
                    job.append_log("Training completed successfully!")
                    job.append_log(f"Final accuracy: {results.get('accuracy', 0):.4f}")
                    job.append_log(f"Final F1: {results.get('f1', 0):.4f}")
                    job.mark_completed(output_dir)

                    # Reload the model in the chatbot service
                    try:
                        from app.services.chatbot_service import chatbot_service
                        job.append_log("Reloading DistilBERT model in chatbot...")
                        chatbot_service.distilbert.reload_model()
                        job.append_log("✓ Model reloaded successfully!")
                    except Exception as e:
                        job.append_log(f"⚠ Failed to reload model: {e}")
                else:
                    job.mark_failed("Training returned no results")

            except Exception as e:
                error_msg = f"Training failed: {str(e)}"
                job.append_log(error_msg)
                job.mark_failed(error_msg)

                import traceback
                job.append_log(traceback.format_exc())

    def start_llama_training(
        self,
        training_file_ids: List[int],
        user_id: int,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ) -> str:
        """
        Start LLaMA training job in background.

        Args:
            training_file_ids: List of training file IDs
            user_id: User ID starting the job
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Job ID
        """
        # Generate job ID
        job_id = f"llama_{uuid.uuid4().hex[:8]}"

        # Create job record
        job = TrainingJob(
            job_id=job_id,
            model_type='llama',
            status='pending',
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_file_ids=','.join(map(str, training_file_ids)),
            started_by=user_id
        )
        db.session.add(job)
        db.session.commit()

        # Get file paths
        training_files = db.session.query(TrainingFile).filter(
            TrainingFile.id.in_(training_file_ids)
        ).all()

        file_paths = [tf.filepath for tf in training_files]

        # Get current app instance to pass to thread
        from flask import current_app
        app = current_app._get_current_object()

        # Start training in background thread
        thread = threading.Thread(
            target=self._run_llama_training,
            args=(app, job.id, file_paths, epochs, batch_size, learning_rate),
            daemon=True
        )
        thread.start()

        self.active_threads[job_id] = thread

        return job_id

    def _run_llama_training(
        self,
        app,
        job_id: int,
        file_paths: List[str],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """
        Run LLaMA training in background.

        Args:
            app: Flask app instance
            job_id: Database job ID
            file_paths: List of training file paths
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        from utils.train_llama import LLaMATrainer

        # Create app context for database access
        with app.app_context():
            # Get job
            job = db.session.get(TrainingJob, job_id)
            if not job:
                return

            try:
                # Mark as running
                job.mark_running()
                job.append_log("LLaMA training job started")
                job.append_log(f"Training files: {len(file_paths)}")
                job.append_log(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

                # Output directory
                output_dir = f"models/llama-{job.job_id}"

                # Check for local LLaMA model path in environment
                local_model_path = os.environ.get('LOCAL_LLAMA_PATH')
                model_name = os.environ.get('LLAMA_MODEL_NAME', 'meta-llama/Llama-3.2-1B-Instruct')

                if local_model_path:
                    job.append_log(f"Using local LLaMA model: {local_model_path}")
                else:
                    job.append_log(f"Using HuggingFace model: {model_name}")

                # Initialize trainer
                trainer = LLaMATrainer(
                    model_name=model_name,
                    output_dir=output_dir,
                    local_model_path=local_model_path
                )

                # Load data
                all_data = []
                for path in file_paths:
                    data = trainer.load_training_data(path)
                    all_data.extend(data)

                job.append_log(f"Loaded {len(all_data)} examples")

                # Calculate total steps
                total_steps = (len(all_data) // batch_size) * epochs
                job.total_steps = total_steps
                db.session.commit()

                # Train
                results = trainer.train(
                    all_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )

                if results:
                    job.append_log("LLaMA training completed successfully!")
                    job.mark_completed(output_dir)
                else:
                    job.mark_failed("LLaMA training returned no results")

            except Exception as e:
                error_msg = f"LLaMA training failed: {str(e)}"
                job.append_log(error_msg)
                job.mark_failed(error_msg)

                import traceback
                job.append_log(traceback.format_exc())

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get training job status.

        Args:
            job_id: Job ID

        Returns:
            Job status dictionary
        """
        job = db.session.query(TrainingJob).filter_by(job_id=job_id).first()

        if not job:
            return {'error': 'Job not found'}

        return job.to_dict()

    def get_active_jobs(self) -> List[Dict]:
        """Get all active training jobs."""
        jobs = TrainingJob.get_active_jobs()
        return [job.to_dict() for job in jobs]

    def get_recent_jobs(self, limit: int = 10) -> List[Dict]:
        """Get recent training jobs."""
        jobs = TrainingJob.get_recent_jobs(limit)
        return [job.to_dict() for job in jobs]


# Global instance
training_service = TrainingService()
