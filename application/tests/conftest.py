"""
PyTest Configuration and Fixtures
"""
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app, db
from app.config import TestingConfig
from app.models.user import User
from app.models.settings import Settings


@pytest.fixture(scope='session')
def app():
    """Create application for testing."""
    app = create_app(TestingConfig)
    return app


@pytest.fixture(scope='function')
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture(scope='function')
def init_database(app):
    """Initialize test database."""
    with app.app_context():
        db.create_all()

        # Create test user
        user = User(username='testuser', email='test@example.com', is_admin=False)
        user.set_password('testpass')
        db.session.add(user)

        # Create admin user
        admin = User(username='admin', email='admin@example.com', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)

        # Create default settings
        settings = Settings(
            auth_required=False,
            chatbot_mode='hybrid',
            use_local_distilbert=True,
            use_local_llama=False
        )
        db.session.add(settings)

        db.session.commit()

        yield db

        db.session.remove()
        db.drop_all()


@pytest.fixture
def authenticated_client(client, init_database):
    """Create authenticated test client."""
    client.post('/login', data={
        'username': 'testuser',
        'password': 'testpass'
    })
    return client


@pytest.fixture
def admin_client(client, init_database):
    """Create authenticated admin client."""
    client.post('/login', data={
        'username': 'admin',
        'password': 'admin123'
    })
    return client
