"""
TrainingFile Model

Manages training data files for LLaMA fine-tuning.
"""
from datetime import datetime
from app import db


class TrainingFile(db.Model):
    """
    Training file model for LLaMA fine-tuning datasets.

    Tracks training data files uploaded for model fine-tuning:
    - Raw datasets
    - Preprocessed/cleaned datasets
    - Example Q&A pairs

    Attributes:
        id: Primary key
        filename: Original filename
        filepath: Storage path on server
        file_type: Category ('raw', 'clean', 'examples')
        dataset_type: Dataset category ('synthetic', 'forum', 'phrasebank')
        num_samples: Number of training samples
        uploaded_by: Foreign key to User
        uploaded_at: Upload timestamp
        processed: Whether file has been preprocessed
        processed_at: When file was processed
        file_size: File size in bytes
        description: Optional description
    """
    __tablename__ = 'training_files'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # 'raw', 'clean', 'examples'
    dataset_type = db.Column(db.String(20), nullable=True)  # 'synthetic', 'forum', 'phrasebank'
    num_samples = db.Column(db.Integer, default=0)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    processed = db.Column(db.Boolean, default=False, nullable=False)
    processed_at = db.Column(db.DateTime, nullable=True)
    file_size = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        """String representation of TrainingFile."""
        return f'<TrainingFile {self.filename} type={self.file_type}>'

    def to_dict(self):
        """
        Convert training file to dictionary for JSON serialization.

        Returns:
            Dictionary representation of training file
        """
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'dataset_type': self.dataset_type,
            'num_samples': self.num_samples,
            'uploaded_by': self.uploaded_by,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'processed': self.processed,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'file_size': self.file_size,
            'description': self.description
        }

    def mark_processed(self, num_samples=0):
        """
        Mark training file as processed.

        Args:
            num_samples: Number of samples in dataset
        """
        self.processed = True
        self.processed_at = datetime.utcnow()
        self.num_samples = num_samples
        db.session.commit()

    @staticmethod
    def get_dataset_stats():
        """
        Get statistics about training datasets.

        Returns:
            Dictionary with dataset statistics
        """
        from sqlalchemy import func

        stats_by_type = db.session.query(
            TrainingFile.dataset_type,
            func.count(TrainingFile.id).label('count'),
            func.sum(TrainingFile.num_samples).label('total_samples')
        ).group_by(TrainingFile.dataset_type).all()

        return {
            'by_dataset_type': {
                row.dataset_type or 'unknown': {
                    'count': row.count,
                    'total_samples': row.total_samples or 0
                }
                for row in stats_by_type
            },
            'total_files': TrainingFile.query.count(),
            'total_samples': db.session.query(func.sum(TrainingFile.num_samples)).scalar() or 0
        }
