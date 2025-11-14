"""
Training Job Model

Tracks model training jobs with real-time progress and metrics.
"""
from datetime import datetime
from app import db


class TrainingJob(db.Model):
    """Model training job with progress tracking."""

    __tablename__ = 'training_jobs'

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    model_type = db.Column(db.String(50), nullable=False)  # 'distilbert' or 'llama'
    status = db.Column(db.String(20), nullable=False, default='pending')  # pending, running, completed, failed

    # Training parameters
    epochs = db.Column(db.Integer, default=3)
    batch_size = db.Column(db.Integer, default=4)
    learning_rate = db.Column(db.Float, default=0.0002)

    # Progress tracking
    current_epoch = db.Column(db.Integer, default=0)
    current_step = db.Column(db.Integer, default=0)
    total_steps = db.Column(db.Integer, default=0)
    progress = db.Column(db.Float, default=0.0)  # 0.0 to 1.0

    # Metrics
    train_loss = db.Column(db.Float)
    eval_loss = db.Column(db.Float)
    accuracy = db.Column(db.Float)

    # Files and output
    training_file_ids = db.Column(db.Text)  # Comma-separated IDs
    output_model_path = db.Column(db.String(500))

    # Metadata
    started_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)

    # Logs
    logs = db.Column(db.Text)  # Training logs

    def __repr__(self):
        return f'<TrainingJob {self.job_id} - {self.status}>'

    def update_progress(self, current_epoch, current_step, total_steps, metrics=None):
        """Update training progress."""
        self.current_epoch = current_epoch
        self.current_step = current_step
        self.total_steps = total_steps
        self.progress = (current_epoch * total_steps + current_step) / (self.epochs * total_steps) if total_steps > 0 else 0

        if metrics:
            if 'loss' in metrics:
                self.train_loss = metrics['loss']
            if 'eval_loss' in metrics:
                self.eval_loss = metrics['eval_loss']
            if 'accuracy' in metrics:
                self.accuracy = metrics['accuracy']

        db.session.commit()

    def mark_running(self):
        """Mark job as running."""
        self.status = 'running'
        self.started_at = datetime.utcnow()
        db.session.commit()

    def mark_completed(self, output_path):
        """Mark job as completed."""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        self.output_model_path = output_path
        self.progress = 1.0
        db.session.commit()

    def mark_failed(self, error_message):
        """Mark job as failed."""
        self.status = 'failed'
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        db.session.commit()

    def append_log(self, message):
        """Append message to logs."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"

        if self.logs:
            self.logs += log_entry
        else:
            self.logs = log_entry

        db.session.commit()

    def to_dict(self):
        """Convert to dictionary for JSON response."""
        return {
            'job_id': self.job_id,
            'model_type': self.model_type,
            'status': self.status,
            'progress': round(self.progress, 4),
            'current_epoch': self.current_epoch,
            'total_epochs': self.epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'metrics': {
                'train_loss': self.train_loss,
                'eval_loss': self.eval_loss,
                'accuracy': self.accuracy
            },
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'output_model_path': self.output_model_path
        }

    @staticmethod
    def get_active_jobs():
        """Get all active (pending/running) jobs."""
        from app import db
        return db.session.query(TrainingJob).filter(
            TrainingJob.status.in_(['pending', 'running'])
        ).all()

    @staticmethod
    def get_recent_jobs(limit=10):
        """Get recent jobs."""
        from app import db
        return db.session.query(TrainingJob).order_by(
            TrainingJob.started_at.desc()
        ).limit(limit).all()
