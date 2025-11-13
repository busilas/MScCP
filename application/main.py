"""
Financial Chatbot - Main Application Entry Point

This module initializes and runs the Flask application.
Provides CLI commands for database initialization and admin user creation.
"""
import os
import sys
import click
from flask.cli import FlaskGroup
from app import create_app, db
from app.models.user import User
from app.models.settings import Settings


# Create Flask app instance
app = create_app()


@app.cli.command('init_db')
def init_database():
    """
    Initialize database tables.

    Creates all tables defined in SQLAlchemy models.
    Also creates default settings if they don't exist.
    """
    with app.app_context():
        click.echo('Creating database tables...')
        db.create_all()

        # Create default settings if not exist
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
            click.echo('✓ Default settings created')

        click.echo('✓ Database initialized successfully')


@app.cli.command('create_admin')
def create_admin():
    """
    Create admin user interactively.

    Prompts for username, email, and password.
    Password is hashed before storage.
    """
    with app.app_context():
        click.echo('\n=== Create Admin User ===\n')

        username = click.prompt('Username', type=str)
        email = click.prompt('Email', type=str)
        password = click.prompt('Password', hide_input=True, confirmation_prompt=True)

        # Check if user exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            click.echo('✗ User with this username or email already exists')
            return

        # Create admin user
        admin = User(
            username=username,
            email=email,
            is_admin=True
        )
        admin.set_password(password)

        db.session.add(admin)
        db.session.commit()

        click.echo(f'\n✓ Admin user "{username}" created successfully')


@app.cli.command('drop_db')
@click.confirmation_option(prompt='Are you sure you want to drop all tables?')
def drop_database():
    """
    Drop all database tables.

    WARNING: This will delete all data!
    """
    with app.app_context():
        click.echo('Dropping all tables...')
        db.drop_all()
        click.echo('✓ All tables dropped')


@app.cli.command('create_default_users')
def create_default_users():
    """
    Create default admin and regular user accounts.

    Creates:
        - Admin: username=admin, password=admin123
        - User: username=user, password=user123
    """
    with app.app_context():
        click.echo('\n=== Creating Default Users ===\n')

        admin, regular_user = User.create_default_users()

        click.echo(f'✓ Admin user created:')
        click.echo(f'  Username: admin')
        click.echo(f'  Email: admin@example.com')
        click.echo(f'  Password: admin123')
        click.echo()
        click.echo(f'✓ Regular user created:')
        click.echo(f'  Username: user')
        click.echo(f'  Email: user@example.com')
        click.echo(f'  Password: user123')
        click.echo()
        click.echo('✓ Default users created successfully')


if __name__ == '__main__':
    # Run Flask development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'

    click.echo('\n' + '='*60)
    click.echo('Financial Chatbot - Starting Application')
    click.echo('='*60)
    click.echo(f'Environment: {os.environ.get("FLASK_ENV", "development")}')
    click.echo(f'Port: {port}')
    click.echo(f'Debug: {debug}')
    click.echo('='*60 + '\n')

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
