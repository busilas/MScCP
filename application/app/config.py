"""
Financial Chatbot - Configuration Module

Defines configuration classes for different environments.
Loads settings from environment variables with sensible defaults.
"""
import os
from datetime import timedelta
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Base configuration class.

    Contains common settings for all environments.
    Environment-specific configs can inherit and override.
    """

    # Flask Core Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')

    # Database Configuration - LOCAL SQLITE
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///financial_chatbot.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True
    }

    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # File Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE', 10 * 1024 * 1024))  # 10MB default
    ALLOWED_EXTENSIONS = set(os.environ.get('ALLOWED_EXTENSIONS', 'pdf,txt,csv').split(','))

    # Model Configuration - DistilBERT
    USE_LOCAL_DISTILBERT = os.environ.get('USE_LOCAL_DISTILBERT', 'True').lower() == 'true'
    DISTILBERT_MODEL_NAME = os.environ.get('DISTILBERT_MODEL_NAME', 'distilbert-base-uncased')
    HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')

    # Model Configuration - LLaMA
    USE_LOCAL_LLAMA = os.environ.get('USE_LOCAL_LLAMA', 'False').lower() == 'true'
    LLAMA_MODEL_NAME = os.environ.get('LLAMA_MODEL_NAME', 'meta-llama/Llama-2-7b-chat-hf')
    LLAMA_API_KEY = os.environ.get('LLAMA_API_KEY')
    LLAMA_API_URL = os.environ.get('LLAMA_API_URL', 'https://api-inference.huggingface.co/models/')

    # FAISS Configuration
    FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'data/faiss_index')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    RAG_TOP_K = int(os.environ.get('RAG_TOP_K', 3))
    RAG_CHUNK_SIZE = int(os.environ.get('RAG_CHUNK_SIZE', 512))

    # Guardian Configuration
    GUARDIAN_ENABLED = os.environ.get('GUARDIAN_ENABLED', 'True').lower() == 'true'
    GUARDIAN_REDACTION_TEXT = os.environ.get('GUARDIAN_REDACTION_TEXT', '[REDACTED]')

    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/app.log')


    # Training Configuration
    TRAINING_FOLDER = os.environ.get('TRAINING_FOLDER', 'training')
    TRAINING_RAW_FOLDER = os.path.join(TRAINING_FOLDER, 'raw')
    TRAINING_CLEAN_FOLDER = os.path.join(TRAINING_FOLDER, 'clean')
    TRAINING_EXAMPLES_FOLDER = os.path.join(TRAINING_FOLDER, 'examples')

    # LLaMA Training Configuration
    LORA_RANK = int(os.environ.get('LORA_RANK', 8))
    LORA_ALPHA = int(os.environ.get('LORA_ALPHA', 32))
    TRAINING_EPOCHS = int(os.environ.get('TRAINING_EPOCHS', 3))
    TRAINING_BATCH_SIZE = int(os.environ.get('TRAINING_BATCH_SIZE', 4))
    TRAINING_LEARNING_RATE = float(os.environ.get('TRAINING_LEARNING_RATE', 2e-4))

    @staticmethod
    def init_app(app):
        """
        Initialize application-specific configuration.

        Args:
            app: Flask application instance
        """
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.FAISS_INDEX_PATH, exist_ok=True)
        os.makedirs(Config.TRAINING_RAW_FOLDER, exist_ok=True)
        os.makedirs(Config.TRAINING_CLEAN_FOLDER, exist_ok=True)
        os.makedirs(Config.TRAINING_EXAMPLES_FOLDER, exist_ok=True)
        os.makedirs('logs', exist_ok=True)


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
