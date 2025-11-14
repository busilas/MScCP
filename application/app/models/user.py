"""
User Model

Defines the User database model with authentication support.
Includes password hashing with bcrypt and Flask-Login integration.
"""
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(UserMixin, db.Model):
    """
    User model for authentication and authorization.

    Attributes:
        id: Primary key
        username: Unique username
        email: Unique email address
        password_hash: Bcrypt hashed password
        is_admin: Boolean flag for admin privileges
        is_active: Boolean flag for account status
        created_at: Timestamp of account creation
        last_login: Timestamp of last login
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relationships
    logs = db.relationship('Log', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    documents = db.relationship('Document', backref='uploaded_by_user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """
        Hash and set user password.

        Args:
            password: Plain text password to hash
        """
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        """
        Verify password against stored hash.

        Args:
            password: Plain text password to verify

        Returns:
            Boolean indicating if password matches
        """
        return check_password_hash(self.password_hash, password)

    def update_last_login(self):
        """Update last login timestamp to current time."""
        self.last_login = datetime.utcnow()
        db.session.commit()

    def __repr__(self):
        """String representation of User."""
        return f'<User {self.username}>'

    def to_dict(self):
        """
        Convert user to dictionary for JSON serialization.

        Returns:
            Dictionary representation of user (excludes password_hash)
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

    @staticmethod
    def create_default_users():
        """
        Create default admin and regular user accounts.

        Creates:
            - Admin user: admin/admin@example.com/admin123
            - Regular user: user/user@example.com/user123

        Returns:
            Tuple of (admin_user, regular_user)
        """
        from app import db

        # Check if users already exist
        admin = db.session.query(User).filter_by(username='admin').first()
        regular_user = db.session.query(User).filter_by(username='user').first()

        # Create admin if doesn't exist
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                is_admin=True,
                is_active=True
            )
            admin.set_password('admin123')
            db.session.add(admin)

        # Create regular user if doesn't exist
        if not regular_user:
            regular_user = User(
                username='user',
                email='user@example.com',
                is_admin=False,
                is_active=True
            )
            regular_user.set_password('user123')
            db.session.add(regular_user)

        db.session.commit()

        return admin, regular_user
