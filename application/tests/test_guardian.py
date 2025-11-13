"""
Tests for Guardian PII Detection
"""
import pytest
from app.services.guardian import Guardian


class TestGuardian:
    """Test Guardian PII detection and redaction."""

    def setup_method(self):
        """Setup test instance."""
        self.guardian = Guardian()

    def test_email_detection(self):
        """Test email address detection."""
        text = "Contact me at john.doe@example.com for more info"
        has_pii, types = self.guardian.detect_pii(text)

        assert has_pii is True
        assert 'email' in types

    def test_phone_detection(self):
        """Test phone number detection."""
        text = "Call me at 555-123-4567 or (555) 987-6543"
        has_pii, types = self.guardian.detect_pii(text)

        assert has_pii is True
        assert 'phone' in types

    def test_ssn_detection(self):
        """Test SSN detection."""
        text = "My SSN is 123-45-6789"
        has_pii, types = self.guardian.detect_pii(text)

        assert has_pii is True
        assert 'ssn' in types

    def test_credit_card_detection(self):
        """Test credit card detection."""
        text = "Card number: 1234-5678-9012-3456"
        has_pii, types = self.guardian.detect_pii(text)

        assert has_pii is True
        assert 'credit_card' in types

    def test_pii_redaction(self):
        """Test PII redaction."""
        text = "Email: test@example.com, Phone: 555-1234"
        redacted = self.guardian.redact_pii(text)

        assert 'test@example.com' not in redacted
        assert '[REDACTED]' in redacted

    def test_clean_text(self):
        """Test text without PII."""
        text = "What is my account balance?"
        has_pii, types = self.guardian.detect_pii(text)

        assert has_pii is False
        assert len(types) == 0

    def test_filter_input(self):
        """Test input filtering."""
        text = "My email is sensitive@example.com"
        result = self.guardian.filter_input(text)

        assert result['has_pii'] is True
        assert result['filtered_text'] != text
        assert 'sensitive@example.com' not in result['filtered_text']

    def test_filter_output(self):
        """Test output filtering."""
        text = "Your account associated with john@example.com is active"
        result = self.guardian.filter_output(text)

        assert result['has_pii'] is True
        assert 'john@example.com' not in result['filtered_text']
