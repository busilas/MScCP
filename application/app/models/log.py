"""
Log Model

Tracks chatbot interactions and performance metrics.
"""
from datetime import datetime
from app import db


class Log(db.Model):
    """
    Logging model for chatbot interactions.

    Records every query with performance metrics:
    - Latency measurements
    - Model mode used
    - Guardian PII detection flags
    - RAG similarity scores

    Attributes:
        id: Primary key
        user_id: Foreign key to User (nullable for anonymous)
        query: User input query
        response: Chatbot response
        mode_used: Model mode ('distilbert', 'llama', 'hybrid')
        latency_ms: Response time in milliseconds
        guardian_flag: Whether Guardian detected/redacted PII
        similarity_score: Best FAISS similarity score for RAG
        timestamp: Query timestamp
        error: Error message if query failed
    """
    __tablename__ = 'logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=True)
    mode_used = db.Column(db.String(20), nullable=False)
    latency_ms = db.Column(db.Integer, nullable=True)
    guardian_flag = db.Column(db.Boolean, default=False, nullable=False)
    similarity_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    error = db.Column(db.Text, nullable=True)

    def __repr__(self):
        """String representation of Log."""
        return f'<Log {self.id} mode={self.mode_used} latency={self.latency_ms}ms>'

    def to_dict(self):
        """
        Convert log to dictionary for JSON serialization.

        Returns:
            Dictionary representation of log
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'query': self.query[:100] + '...' if len(self.query) > 100 else self.query,
            'response': self.response[:100] + '...' if self.response and len(self.response) > 100 else self.response,
            'mode_used': self.mode_used,
            'latency_ms': self.latency_ms,
            'guardian_flag': self.guardian_flag,
            'similarity_score': self.similarity_score,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'error': self.error
        }

    @staticmethod
    def get_performance_stats():
        """
        Calculate aggregate performance statistics.

        Returns:
            Dictionary with performance metrics:
            - avg_latency_by_mode: Average latency per model mode
            - request_count_by_mode: Request counts per mode
            - guardian_redaction_rate: Percentage of queries with PII
            - avg_similarity_score: Average RAG similarity
        """
        from sqlalchemy import func

        # Average latency by mode
        latency_by_mode = db.session.query(
            Log.mode_used,
            func.avg(Log.latency_ms).label('avg_latency'),
            func.count(Log.id).label('count')
        ).group_by(Log.mode_used).all()

        # Guardian statistics
        total_queries = Log.query.count()
        guardian_flagged = Log.query.filter_by(guardian_flag=True).count()
        guardian_rate = (guardian_flagged / total_queries * 100) if total_queries > 0 else 0

        # Average similarity score
        avg_similarity = db.session.query(
            func.avg(Log.similarity_score)
        ).filter(Log.similarity_score.isnot(None)).scalar() or 0.0

        return {
            'avg_latency_by_mode': {row.mode_used: float(row.avg_latency or 0) for row in latency_by_mode},
            'request_count_by_mode': {row.mode_used: row.count for row in latency_by_mode},
            'guardian_redaction_rate': round(guardian_rate, 2),
            'avg_similarity_score': round(float(avg_similarity), 4),
            'total_queries': total_queries
        }
