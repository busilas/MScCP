"""
Tests for API Endpoints
"""
import pytest
import json


class TestChatAPI:
    """Test chat API endpoints."""

    def test_query_endpoint(self, client, init_database):
        """Test chat query endpoint."""
        response = client.post('/api/query',
            data=json.dumps({'query': 'What is my account balance?'}),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'response' in data
        assert 'mode_used' in data

    def test_empty_query(self, client, init_database):
        """Test empty query rejection."""
        response = client.post('/api/query',
            data=json.dumps({'query': ''}),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_missing_query(self, client, init_database):
        """Test missing query parameter."""
        response = client.post('/api/query',
            data=json.dumps({}),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_history_endpoint(self, authenticated_client, init_database):
        """Test history endpoint."""
        response = authenticated_client.get('/api/history')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'history' in data


class TestAdminAPI:
    """Test admin API endpoints."""

    def test_update_settings(self, admin_client, init_database):
        """Test settings update."""
        response = admin_client.post('/admin/settings',
            data=json.dumps({
                'auth_required': True,
                'chatbot_mode': 'hybrid'
            }),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_update_settings_invalid_mode(self, admin_client, init_database):
        """Test invalid mode rejection."""
        response = admin_client.post('/admin/settings',
            data=json.dumps({
                'chatbot_mode': 'invalid_mode'
            }),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_performance_data(self, admin_client, init_database):
        """Test performance data endpoint."""
        response = admin_client.get('/admin/api/performance/data')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'avg_latency_by_mode' in data
        assert 'request_count_by_mode' in data
