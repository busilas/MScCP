"""
Authentication Routes

Handles user login, logout, and registration.
"""
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models.user import User
from app.models.settings import Settings


auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    User login page and handler.

    GET: Display login form
    POST: Process login credentials
    """
    # Check if auth is required
    settings = Settings.get_settings()

    if current_user.is_authenticated:
        return redirect(url_for('chat.index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            if not user.is_active:
                flash('Account is disabled. Please contact support.', 'error')
                return redirect(url_for('auth.login'))

            login_user(user, remember=remember)
            user.update_last_login()

            flash(f'Welcome back, {user.username}!', 'success')

            next_page = request.args.get('next')
            return redirect(next_page or url_for('chat.index'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html', auth_required=settings.auth_required)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration page and handler.

    GET: Display registration form
    POST: Create new user account
    """
    if current_user.is_authenticated:
        return redirect(url_for('chat.index'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('All fields are required.', 'error')
            return redirect(url_for('auth.register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('auth.register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return redirect(url_for('auth.register'))

        # Check if user exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            flash('Username or email already exists.', 'error')
            return redirect(url_for('auth.register'))

        # Create new user
        new_user = User(
            username=username,
            email=email,
            is_admin=False
        )
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Log out current user."""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('auth.login'))
