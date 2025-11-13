"""
Tests for Authentication
"""
import pytest


class TestAuthentication:
    """Test user authentication."""

    def test_login_page(self, client):
        """Test login page loads."""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'Login' in response.data

    def test_register_page(self, client):
        """Test register page loads."""
        response = client.get('/register')
        assert response.status_code == 200
        assert b'Register' in response.data

    def test_successful_login(self, client, init_database):
        """Test successful login."""
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'testpass'
        }, follow_redirects=True)

        assert response.status_code == 200

    def test_invalid_login(self, client, init_database):
        """Test invalid login credentials."""
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'wrongpassword'
        }, follow_redirects=True)

        assert b'Invalid' in response.data or b'incorrect' in response.data.lower()

    def test_successful_registration(self, client, init_database):
        """Test user registration."""
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'newpass123',
            'confirm_password': 'newpass123'
        }, follow_redirects=True)

        assert response.status_code == 200

    def test_duplicate_username(self, client, init_database):
        """Test duplicate username rejection."""
        response = client.post('/register', data={
            'username': 'testuser',  # Already exists
            'email': 'another@example.com',
            'password': 'pass123',
            'confirm_password': 'pass123'
        }, follow_redirects=True)

        assert b'already exists' in response.data.lower()

    def test_password_mismatch(self, client, init_database):
        """Test password confirmation."""
        response = client.post('/register', data={
            'username': 'newuser2',
            'email': 'new2@example.com',
            'password': 'pass123',
            'confirm_password': 'different'
        }, follow_redirects=True)

        assert b'match' in response.data.lower()

    def test_logout(self, authenticated_client):
        """Test user logout."""
        response = authenticated_client.get('/logout', follow_redirects=True)
        assert response.status_code == 200
