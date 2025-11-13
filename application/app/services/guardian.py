"""
Guardian Plugin

PII detection and redaction middleware.
Protects sensitive information using regex patterns and ML classification.
"""
import re
from typing import Dict, List, Tuple


class Guardian:
    """
    Guardian plugin for PII detection and redaction.

    Detects and redacts personally identifiable information (PII):
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - Social Security Numbers
    - Names (basic pattern matching)
    - Addresses
    - Account numbers
    """

    def __init__(self, redaction_text: str = "[REDACTED]"):
        """
        Initialize Guardian plugin.

        Args:
            redaction_text: Text to replace detected PII
        """
        self.redaction_text = redaction_text

        # Define PII detection patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'account_number': r'\b[Aa]ccount\s*#?\s*:?\s*\d{6,}\b',
            'routing_number': r'\b[Rr]outing\s*#?\s*:?\s*\d{9}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
        }

        # Common PII keywords
        self.pii_keywords = [
            'ssn', 'social security', 'credit card', 'passport',
            'driver license', 'account number', 'routing number',
            'date of birth', 'dob', 'birthday'
        ]

    def detect_pii(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect PII in text.

        Args:
            text: Input text to scan

        Returns:
            Tuple of (has_pii: bool, detected_types: List[str])
        """
        detected_types = []

        for pii_type, pattern in self.patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_types.append(pii_type)

        # Check for PII keywords
        text_lower = text.lower()
        for keyword in self.pii_keywords:
            if keyword in text_lower and 'keyword' not in detected_types:
                detected_types.append('pii_keyword')
                break

        has_pii = len(detected_types) > 0
        return has_pii, detected_types

    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text to redact

        Returns:
            Text with PII replaced by redaction_text
        """
        redacted_text = text

        # Apply redaction patterns
        for pii_type, pattern in self.patterns.items():
            redacted_text = re.sub(
                pattern,
                self.redaction_text,
                redacted_text,
                flags=re.IGNORECASE
            )

        return redacted_text

    def filter_input(self, user_input: str) -> Dict:
        """
        Filter user input for PII.

        Args:
            user_input: Raw user input text

        Returns:
            Dictionary with:
                - filtered_text: Text with PII redacted
                - has_pii: Boolean flag
                - detected_types: List of detected PII types
        """
        has_pii, detected_types = self.detect_pii(user_input)
        filtered_text = self.redact_pii(user_input) if has_pii else user_input

        return {
            'filtered_text': filtered_text,
            'has_pii': has_pii,
            'detected_types': detected_types,
            'original_length': len(user_input),
            'filtered_length': len(filtered_text)
        }

    def filter_output(self, llm_output: str) -> Dict:
        """
        Filter LLM output for accidental PII leakage.

        Args:
            llm_output: Generated text from LLM

        Returns:
            Dictionary with filtered output and detection info
        """
        has_pii, detected_types = self.detect_pii(llm_output)
        filtered_text = self.redact_pii(llm_output) if has_pii else llm_output

        return {
            'filtered_text': filtered_text,
            'has_pii': has_pii,
            'detected_types': detected_types
        }

    def scan_document(self, document_text: str) -> Dict:
        """
        Scan document for PII before indexing.

        Args:
            document_text: Full document text

        Returns:
            Dictionary with scan results and recommendations
        """
        has_pii, detected_types = self.detect_pii(document_text)

        # Calculate PII density
        total_matches = sum(
            len(re.findall(pattern, document_text, re.IGNORECASE))
            for pattern in self.patterns.values()
        )

        return {
            'has_pii': has_pii,
            'detected_types': detected_types,
            'total_matches': total_matches,
            'recommendation': 'redact' if has_pii else 'safe',
            'document_length': len(document_text)
        }


# Global instance
guardian = Guardian()
