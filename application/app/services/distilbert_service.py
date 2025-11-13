"""
DistilBERT Service

Intent detection and entity extraction using DistilBERT.
Supports both local model and API-based inference.
"""
import os
import re
from typing import Dict, List, Optional
import requests


class DistilBERTService:
    """
    DistilBERT service for intent detection and entity extraction.

    Capabilities:
    - Intent classification
    - Named entity recognition
    - Financial entity extraction (tickers, currencies, dates)
    """

    def __init__(self, use_local: bool = True, api_key: Optional[str] = None):
        """
        Initialize DistilBERT service.

        Args:
            use_local: Whether to use local model or API
            api_key: HuggingFace API key for hosted inference
        """
        self.use_local = use_local
        self.api_key = api_key
        self.model_name = "distilbert-base-uncased"

        # Financial intents
        self.intents = [
            'account_inquiry',
            'transaction_history',
            'balance_check',
            'payment',
            'investment_advice',
            'loan_inquiry',
            'market_data',
            'general_banking',
            'complaint',
            'other'
        ]

        # Load local model if needed
        if self.use_local:
            self._load_local_model()

    def _load_local_model(self):
        """Load local DistilBERT model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.intents)
            )
            print(f"✓ Local DistilBERT model loaded: {self.model_name}")
        except ImportError:
            print("⚠ Transformers library not installed. Install with: pip install transformers torch")
            self.use_local = False
        except Exception as e:
            print(f"⚠ Failed to load local model: {e}")
            self.use_local = False

    def detect_intent(self, text: str) -> Dict:
        """
        Detect intent from user query.

        Args:
            text: User query text

        Returns:
            Dictionary with intent and confidence score
        """
        if self.use_local:
            return self._detect_intent_local(text)
        else:
            return self._detect_intent_api(text)

    def _detect_intent_local(self, text: str) -> Dict:
        """Detect intent using local model."""
        try:
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(predictions, dim=1)

            intent = self.intents[predicted_class.item()]
            confidence_score = confidence.item()

            return {
                'intent': intent,
                'confidence': round(confidence_score, 4),
                'all_scores': {
                    intent: round(score, 4)
                    for intent, score in zip(self.intents, predictions[0].tolist())
                }
            }
        except Exception as e:
            print(f"Error in local intent detection: {e}")
            return self._detect_intent_fallback(text)

    def _detect_intent_api(self, text: str) -> Dict:
        """Detect intent using HuggingFace API."""
        if not self.api_key:
            return self._detect_intent_fallback(text)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": text},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    'intent': result[0]['label'] if result else 'general_banking',
                    'confidence': result[0]['score'] if result else 0.5
                }
            else:
                return self._detect_intent_fallback(text)
        except Exception as e:
            print(f"Error in API intent detection: {e}")
            return self._detect_intent_fallback(text)

    def _detect_intent_fallback(self, text: str) -> Dict:
        """Fallback intent detection using keyword matching."""
        text_lower = text.lower()

        # Keyword-based intent detection
        if any(word in text_lower for word in ['balance', 'how much', 'account balance']):
            return {'intent': 'balance_check', 'confidence': 0.7}
        elif any(word in text_lower for word in ['transaction', 'history', 'statement']):
            return {'intent': 'transaction_history', 'confidence': 0.7}
        elif any(word in text_lower for word in ['transfer', 'send', 'payment', 'pay']):
            return {'intent': 'payment', 'confidence': 0.7}
        elif any(word in text_lower for word in ['loan', 'mortgage', 'credit']):
            return {'intent': 'loan_inquiry', 'confidence': 0.7}
        elif any(word in text_lower for word in ['invest', 'stock', 'market', 'portfolio']):
            return {'intent': 'investment_advice', 'confidence': 0.7}
        elif any(word in text_lower for word in ['complaint', 'issue', 'problem']):
            return {'intent': 'complaint', 'confidence': 0.7}
        else:
            return {'intent': 'general_banking', 'confidence': 0.5}

    def extract_entities(self, text: str) -> Dict:
        """
        Extract financial entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        entities = {
            'tickers': self._extract_tickers(text),
            'currencies': self._extract_currencies(text),
            'dates': self._extract_dates(text),
            'amounts': self._extract_amounts(text)
        }

        return entities

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers (e.g., AAPL, GOOGL)."""
        pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(pattern, text)

        # Filter out common words
        common_words = {'USD', 'EUR', 'GBP', 'CAD', 'THE', 'AND', 'FOR', 'ARE'}
        return [t for t in potential_tickers if t not in common_words]

    def _extract_currencies(self, text: str) -> List[str]:
        """Extract currency codes (e.g., USD, EUR)."""
        pattern = r'\b(?:USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY|INR)\b'
        return re.findall(pattern, text, re.IGNORECASE)

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date patterns."""
        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]

        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return dates

    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts."""
        pattern = r'[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|EUR|GBP)'
        return re.findall(pattern, text, re.IGNORECASE)

    def process_query(self, query: str) -> Dict:
        """
        Comprehensive query processing.

        Args:
            query: User query

        Returns:
            Dictionary with intent and entities
        """
        intent_result = self.detect_intent(query)
        entities = self.extract_entities(query)

        return {
            'intent': intent_result['intent'],
            'confidence': intent_result.get('confidence', 0.5),
            'entities': entities,
            'query_length': len(query)
        }
