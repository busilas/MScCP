"""
LLaMA Service

Natural language generation using LLaMA.
Supports both local model and API-based inference.
Uses advanced prompt engineering from uploaded document examples.
"""
import os
import requests
from typing import Dict, List, Optional
from app.services.prompt_engineering import prompt_engineer


class LLaMAService:
    """
    LLaMA service for natural language generation.

    Provides conversational responses for financial queries
    with context from RAG and DistilBERT entities.
    """

    def __init__(self, use_local: bool = False, api_key: Optional[str] = None):
        """
        Initialize LLaMA service.

        Args:
            use_local: Whether to use local model or API
            api_key: API key for hosted inference
        """
        self.use_local = use_local
        self.api_key = api_key
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"

        # System prompt for financial assistant
        self.system_prompt = """You are a helpful and professional financial assistant.
You provide accurate, clear, and concise information about banking, investments, and financial services.
You always maintain user privacy and never ask for sensitive personal information.
If you don't know something, you honestly say so rather than making up information."""

        # Load local model if needed
        if self.use_local:
            self._load_local_model()

    def _load_local_model(self):
        """Load local LLaMA model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            print(f"Loading LLaMA model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print("✓ Local LLaMA model loaded")
        except ImportError:
            print("⚠ Transformers library not installed")
            self.use_local = False
        except Exception as e:
            print(f"⚠ Failed to load local LLaMA model: {e}")
            print("  Note: LLaMA requires significant memory (>16GB RAM recommended)")
            self.use_local = False

    def generate_response(
        self,
        query: str,
        context: Optional[List[str]] = None,
        entities: Optional[Dict] = None,
        intent: Optional[str] = None,
        max_length: int = 512,
        prompt_style: str = 'secure_default',
        use_few_shot: bool = True,
        use_chain_of_thought: bool = True
    ) -> Dict:
        """
        Generate response to user query with advanced prompt engineering.
        Uses examples from uploaded financial chatbot document.

        Args:
            query: User query
            context: Retrieved documents from RAG
            entities: Extracted entities from DistilBERT
            intent: Detected intent (for context-aware prompting)
            max_length: Maximum response length
            prompt_style: Prompt style (secure_default/compliance_strict/educational/customer_service)
            use_few_shot: Include few-shot financial examples
            use_chain_of_thought: Enable chain-of-thought reasoning

        Returns:
            Dictionary with generated response and metadata
        """
        # Build prompt using advanced prompt engineering module
        prompt = prompt_engineer.build_prompt(
            query=query,
            context=context,
            entities=entities,
            intent=intent,
            style=prompt_style,
            use_few_shot=use_few_shot,
            use_chain_of_thought=use_chain_of_thought,
            include_security_reminder=True
        )

        if self.use_local:
            return self._generate_local(prompt, max_length)
        else:
            return self._generate_api(prompt, max_length)

    def _build_prompt(
        self,
        query: str,
        context: Optional[List[str]] = None,
        entities: Optional[Dict] = None
    ) -> str:
        """
        Build prompt with system instructions, context, and entities.

        Args:
            query: User query
            context: Retrieved documents
            entities: Extracted entities

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n"
        ]

        # Add context from RAG
        if context and len(context) > 0:
            prompt_parts.append("\nRelevant Information:\n")
            for i, doc in enumerate(context, 1):
                prompt_parts.append(f"{i}. {doc}\n")

        # Add extracted entities
        if entities:
            entity_str = self._format_entities(entities)
            if entity_str:
                prompt_parts.append(f"\nDetected Entities: {entity_str}\n")

        # Add user query
        prompt_parts.append(f"\nUser Query: {query}\n")
        prompt_parts.append("\nPlease provide a helpful and accurate response. [/INST]")

        return "".join(prompt_parts)

    def _format_entities(self, entities: Dict) -> str:
        """Format entities for prompt."""
        parts = []

        if entities.get('tickers'):
            parts.append(f"Stocks: {', '.join(entities['tickers'])}")
        if entities.get('currencies'):
            parts.append(f"Currencies: {', '.join(entities['currencies'])}")
        if entities.get('amounts'):
            parts.append(f"Amounts: {', '.join(entities['amounts'][:3])}")
        if entities.get('dates'):
            parts.append(f"Dates: {', '.join(entities['dates'][:2])}")

        return "; ".join(parts)

    def _generate_local(self, prompt: str, max_length: int) -> Dict:
        """Generate response using local model."""
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

            return {
                'response': response,
                'model': 'local',
                'tokens_generated': outputs[0].shape[0] - inputs['input_ids'].shape[1]
            }
        except Exception as e:
            print(f"Error in local generation: {e}")
            return self._generate_fallback(prompt)

    def _generate_api(self, prompt: str, max_length: int) -> Dict:
        """Generate response using API."""
        if not self.api_key:
            return self._generate_fallback(prompt)

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "return_full_text": False
                }
            }

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')

                return {
                    'response': generated_text.strip(),
                    'model': 'api',
                    'api_status': 'success'
                }
            else:
                print(f"API error: {response.status_code}")
                return self._generate_fallback(prompt)
        except Exception as e:
            print(f"Error in API generation: {e}")
            return self._generate_fallback(prompt)

    def _generate_fallback(self, prompt: str) -> Dict:
        """Fallback response when model unavailable."""
        return {
            'response': "I apologize, but I'm currently unable to generate a detailed response. "
                       "This could be due to high demand or system maintenance. "
                       "Please try again in a moment, or contact support if the issue persists.",
            'model': 'fallback',
            'note': 'LLaMA model unavailable'
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 512
    ) -> Dict:
        """
        Multi-turn chat conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum response length

        Returns:
            Dictionary with response
        """
        # Build conversation prompt
        prompt_parts = [f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n"]

        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                prompt_parts.append(f"{content} [/INST] ")
            elif role == 'assistant':
                prompt_parts.append(f"{content} </s><s>[INST] ")

        prompt = "".join(prompt_parts)

        if self.use_local:
            return self._generate_local(prompt, max_length)
        else:
            return self._generate_api(prompt, max_length)
