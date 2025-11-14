"""
Financial Chatbot - Application Factory

This module implements the Flask application factory pattern.
Creates and configures the Flask app with all extensions and blueprints.
"""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_cors import CORS
from app.config import Config


# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()


def create_app(config_class=Config):
    """
    Application factory function.

    Creates and configures a Flask application instance with all necessary
    extensions, blueprints, and error handlers.

    Args:
        config_class: Configuration class to use (default: Config)

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    CORS(app)

    # Configure Flask-Login
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    # User loader for Flask-Login
    from app.models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        """Load user by ID for Flask-Login."""
        return db.session.get(User, int(user_id))

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.chat import chat_bp
    from app.routes.admin import admin_bp
    from app.routes.upload import upload_bp
    from app.routes.training import training_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(upload_bp, url_prefix='/upload')
    app.register_blueprint(training_bp, url_prefix='/training')

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors."""
        return {'error': 'Resource not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        db.session.rollback()
        return {'error': 'Internal server error'}, 500

    # Context processor for templates
    @app.context_processor
    def inject_settings():
        """Inject settings into all templates."""
        from app.models.settings import Settings
        settings = db.session.query(Settings).first()
        return {'app_settings': settings}

    return app
