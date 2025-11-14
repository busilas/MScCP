"""
Chatbot Service

Orchestrates DistilBERT, LLaMA, RAG, and Guardian services
to provide comprehensive chatbot responses.
"""
import time
from typing import Dict, Optional
from app.services.guardian import guardian
from app.services.distilbert_service import DistilBERTService
from app.services.llama_service import LLaMAService
from app.services.rag_service import RAGService
from app.models.settings import Settings


class ChatbotService:
    """
    Main chatbot service orchestrator.

    Coordinates all AI services based on configured mode:
    - DistilBERT-only: Intent detection with rule-based responses
    - LLaMA-only: Direct LLM generation
    - Hybrid: Combined DistilBERT + LLaMA + RAG
    """

    def __init__(self):
        """Initialize chatbot service with all components."""
        self.distilbert = None
        self.llama = None
        self.rag = None
        self.guardian = guardian
        self._initialized = False

    def _initialize_services(self):
        """Initialize AI services based on settings (lazy initialization)."""
        if self._initialized:
            return

        from flask import current_app

        # Get settings from database (with app context)
        settings = Settings.get_settings()

        # Initialize DistilBERT
        self.distilbert = DistilBERTService(
            use_local=settings.use_local_distilbert
        )

        # Initialize LLaMA (pass API key from environment)
        import os
        api_key = os.environ.get('HUGGINGFACE_API_KEY')
        self.llama = LLaMAService(
            use_local=settings.use_local_llama,
            api_key=api_key
        )

        # Initialize RAG
        self.rag = RAGService()

        self._initialized = True

    def process_query(
        self,
        query: str,
        mode: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict:
        """
        Process user query through chatbot pipeline.

        Args:
            query: User query text
            mode: Override mode ('distilbert', 'llama', 'hybrid')
            user_id: User ID for logging

        Returns:
            Dictionary with response and metadata
        """
        # Lazy initialize services
        self._initialize_services()

        start_time = time.time()

        # Get mode from settings if not specified
        if mode is None:
            settings = Settings.get_settings()
            mode = settings.chatbot_mode

        # Step 1: Guardian input filtering
        guardian_input = self.guardian.filter_input(query)
        filtered_query = guardian_input['filtered_text']
        has_pii = guardian_input['has_pii']

        # Step 2: Process based on mode
        if mode == 'distilbert':
            result = self._process_distilbert_only(filtered_query)
        elif mode == 'llama':
            result = self._process_llama_only(filtered_query)
        else:  # hybrid
            result = self._process_hybrid(filtered_query)

        # Step 3: Guardian output filtering
        guardian_output = self.guardian.filter_output(result['response'])
        final_response = guardian_output['filtered_text']

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Compile final result
        return {
            'response': final_response,
            'mode_used': mode,
            'latency_ms': latency_ms,
            'guardian_flag': has_pii or guardian_output['has_pii'],
            'similarity_score': result.get('similarity_score'),
            'intent': result.get('intent'),
            'entities': result.get('entities'),
            'metadata': {
                'pii_detected_input': has_pii,
                'pii_detected_output': guardian_output['has_pii'],
                'rag_used': result.get('rag_used', False),
                'num_context_docs': result.get('num_context_docs', 0)
            }
        }

    def _process_distilbert_only(self, query: str) -> Dict:
        """Process query using only DistilBERT."""
        # Detect intent and extract entities
        analysis = self.distilbert.process_query(query)

        # Generate rule-based response
        response = self._generate_rule_based_response(
            analysis['intent'],
            analysis['entities']
        )

        return {
            'response': response,
            'intent': analysis['intent'],
            'entities': analysis['entities'],
            'rag_used': False
        }

    def _process_llama_only(self, query: str) -> Dict:
        """Process query using only LLaMA (falls back to DistilBERT if unavailable)."""
        # Try LLaMA first, but fall back to DistilBERT if it fails
        try:
            llama_result = self.llama.generate_response(query)

            # If we got a fallback response, use DistilBERT instead
            if 'unable to generate' in llama_result['response'].lower():
                return self._process_distilbert_only(query)

            return {
                'response': llama_result['response'],
                'rag_used': False
            }
        except Exception:
            # Fall back to DistilBERT on any error
            return self._process_distilbert_only(query)

    def _process_hybrid(self, query: str) -> Dict:
        """Process query using hybrid approach (falls back to DistilBERT if LLaMA unavailable)."""
        # Step 1: DistilBERT analysis
        analysis = self.distilbert.process_query(query)

        # Step 2: RAG retrieval
        context_docs, similarity_score = self.rag.get_context(query, top_k=3)

        # Step 3: Try LLaMA, fall back to DistilBERT if unavailable
        try:
            llama_result = self.llama.generate_response(
                query,
                context=context_docs,
                entities=analysis['entities'],
                intent=analysis['intent'],
                use_few_shot=True,  # Enable examples from uploaded document
                use_chain_of_thought=True  # Enable step-by-step reasoning
            )

            # If LLaMA returns fallback message, use DistilBERT instead
            if 'unable to generate' in llama_result['response'].lower():
                response = self._generate_rule_based_response(
                    analysis['intent'],
                    analysis['entities']
                )
            else:
                response = llama_result['response']
        except Exception:
            # Fall back to DistilBERT on any error
            response = self._generate_rule_based_response(
                analysis['intent'],
                analysis['entities']
            )

        return {
            'response': response,
            'intent': analysis['intent'],
            'entities': analysis['entities'],
            'similarity_score': similarity_score,
            'rag_used': len(context_docs) > 0,
            'num_context_docs': len(context_docs)
        }

    def _generate_rule_based_response(self, intent: str, entities: Dict) -> str:
        """
        Generate rule-based response for DistilBERT-only mode.

        Args:
            intent: Detected intent
            entities: Extracted entities

        Returns:
            Rule-based response string
        """
        responses = {
            'balance_check': "To check your account balance, please log into your online banking portal or mobile app. "
                           "You can also call our customer service at 1-800-BANK-123.",

            'transaction_history': "You can view your transaction history by logging into your account online or through our mobile app. "
                                  "Statements are available for the past 12 months.",

            'payment': "To make a payment, you can use our online banking, mobile app, or set up automatic payments. "
                      "Payments typically process within 1-2 business days.",

            'loan_inquiry': "We offer various loan products including personal loans, mortgages, and auto loans. "
                          "You can apply online or schedule an appointment with a loan officer.",

            'investment_advice': "For investment advice, I recommend speaking with one of our certified financial advisors. "
                               "You can schedule a consultation through our website or by calling 1-800-INVEST.",

            'complaint': "I'm sorry you're experiencing an issue. Please contact our customer service team at 1-800-SUPPORT "
                       "or submit a formal complaint through our website. We take all concerns seriously.",

            'general_banking': "I'm here to help with your banking questions. You can ask me about accounts, transactions, "
                             "loans, investments, or any other banking services."
        }

        response = responses.get(intent, responses['general_banking'])

        # Add entity information if available
        if entities.get('amounts'):
            response += f" I noticed you mentioned amounts: {', '.join(entities['amounts'][:2])}."

        if entities.get('tickers'):
            response += f" Regarding {', '.join(entities['tickers'])}, please check our investment platform for current quotes."

        return response

    def add_document_to_rag(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add document to RAG index.

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            Number of chunks added
        """
        # Lazy initialize services
        self._initialize_services()

        return self.rag.add_document(text, metadata)

    def get_rag_stats(self) -> Dict:
        """Get RAG index statistics."""
        # Lazy initialize services
        self._initialize_services()

        return self.rag.get_stats()


# Global instance
chatbot_service = ChatbotService()
