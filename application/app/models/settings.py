"""
Settings Model

Defines global application settings that control chatbot behavior.
"""
from datetime import datetime
from app import db


class Settings(db.Model):
    """
    Application settings model.

    Stores configurable settings that affect chatbot behavior:
    - Authentication requirements
    - Chatbot mode (DistilBERT, LLaMA, or Hybrid)
    - Model execution mode (local vs API)

    Attributes:
        id: Primary key
        auth_required: Whether authentication is required to use chatbot
        chatbot_mode: Operating mode ('distilbert', 'llama', 'hybrid')
        use_local_distilbert: Use local DistilBERT model vs API
        use_local_llama: Use local LLaMA model vs API
        updated_at: Last update timestamp
    """
    __tablename__ = 'settings'

    id = db.Column(db.Integer, primary_key=True)
    auth_required = db.Column(db.Boolean, default=False, nullable=False)
    chatbot_mode = db.Column(db.String(20), default='hybrid', nullable=False)
    use_local_distilbert = db.Column(db.Boolean, default=True, nullable=False)
    use_local_llama = db.Column(db.Boolean, default=False, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        """String representation of Settings."""
        return f'<Settings mode={self.chatbot_mode} auth={self.auth_required}>'

    def to_dict(self):
        """
        Convert settings to dictionary for JSON serialization.

        Returns:
            Dictionary representation of settings
        """
        return {
            'id': self.id,
            'auth_required': self.auth_required,
            'chatbot_mode': self.chatbot_mode,
            'use_local_distilbert': self.use_local_distilbert,
            'use_local_llama': self.use_local_llama,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @staticmethod
    def get_settings():
        """
        Get current settings (singleton pattern).

        Returns:
            Settings instance, creates default if none exists
        """
        settings = Settings.query.first()
        if not settings:
            settings = Settings(
                auth_required=False,
                chatbot_mode='hybrid',
                use_local_distilbert=True,
                use_local_llama=False
            )
            db.session.add(settings)
            db.session.commit()
        return settings

    def update_settings(self, **kwargs):
        """
        Update settings with provided keyword arguments.

        Args:
            **kwargs: Settings fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        db.session.commit()
