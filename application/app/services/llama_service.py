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
        import os
        self.use_local = use_local

        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get('HUGGINGFACE_API_KEY')

        # Check for local model path first
        local_model_path = os.environ.get('LOCAL_LLAMA_PATH', '')
        if local_model_path and os.path.exists(local_model_path) and self.use_local:
            self.model_name = local_model_path
            print(f"✓ Using local LLaMA model from: {local_model_path}")
        else:
            self.model_name = os.environ.get('LLAMA_MODEL_NAME', 'meta-llama/Llama-3.2-3B-Instruct')
            if self.use_local:
                print(f"⚠ Local model requested but path not found: {local_model_path}")
                print(f"  Falling back to API mode with model: {self.model_name}")
                self.use_local = False
            else:
                print(f"✓ Using LLaMA API mode with model: {self.model_name}")

        # Show API key status
        if not self.use_local:
            if self.api_key and self.api_key != 'your-huggingface-api-key':
                print(f"✓ Hugging Face API key found (starts with: {self.api_key[:7]}...)")
            else:
                print("⚠ WARNING: No valid Hugging Face API key found!")
                print("  Get your free key: https://huggingface.co/settings/tokens")
                print("  Add it to .env file: HUGGINGFACE_API_KEY=hf_your_key_here")

        # System prompt for financial assistant
        self.system_prompt = """You are a helpful and professional financial assistant.
You provide accurate, clear, and concise information about banking, investments, and financial services.
You always maintain user privacy and never ask for sensitive personal information.
If you don't know something, you honestly say so rather than making up information."""

        # Load local model if needed
        if self.use_local:
            self._load_local_model()

    def _load_local_model(self):
        """Load local LLaMA model with 8-bit quantization."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            print(f"Loading LLaMA model: {self.model_name}")
            print("Using 8-bit quantization to reduce GPU memory usage...")

            # Configure 8-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("✓ Local LLaMA model loaded successfully with 8-bit quantization")
            print(f"✓ GPU Memory saved: ~40-50% compared to full precision")
        except ImportError as e:
            print(f"⚠ Required library not installed: {e}")
            print("  Install with: pip install bitsandbytes accelerate")
            self.use_local = False
        except Exception as e:
            print(f"⚠ Failed to load local LLaMA model: {e}")
            print("  Note: Requires CUDA-compatible GPU with 10GB+ VRAM")
            print("  Trying fallback without quantization...")
            try:
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                print("✓ Loaded without quantization (requires more GPU memory)")
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2}")
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
        # Skip API calls if model is disabled
        if self.model_name == 'disabled':
            return self._generate_fallback(query)

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

            # Check if model is loaded
            if not hasattr(self, 'model') or self.model is None:
                print("✗ Local model not loaded - falling back")
                return self._generate_fallback(prompt)

            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                print("✗ Tokenizer not loaded - falling back")
                return self._generate_fallback(prompt)

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
            import traceback
            print(f"✗ Error in local generation: {e}")
            print(f"✗ Traceback: {traceback.format_exc()}")
            return self._generate_fallback(prompt)

    def _generate_api(self, prompt: str, max_length: int) -> Dict:
        """Generate response using Hugging Face Inference API or Ollama."""

        # Try Groq first (free, fast LLaMA)
        groq_result = self._try_groq(prompt, max_length)
        if groq_result:
            return groq_result

        # Try OpenAI second (if key available)
        openai_result = self._try_openai(prompt, max_length)
        if openai_result:
            return openai_result

        # Try Ollama third (local, fast, free)
        ollama_result = self._try_ollama(prompt, max_length)
        if ollama_result:
            return ollama_result

        # Fall back to Hugging Face API
        if not self.api_key or self.api_key == 'your-huggingface-api-key':
            print("✗ No valid API key found")
            return self._generate_fallback(prompt)

        try:
            from huggingface_hub import InferenceClient

            print(f"📡 Calling Hugging Face API: {self.model_name}")
            client = InferenceClient(token=self.api_key)

            free_models = [
                ("google/flan-t5-base", "text"),
                ("bigscience/bloom-560m", "text"),
                ("gpt2", "text"),
                ("distilgpt2", "text"),
                ("EleutherAI/gpt-neo-125m", "text")
            ]

            models_to_try = [(self.model_name, "text")] + free_models

            for model_name, mode in models_to_try:
                try:
                    print(f"   Trying: {model_name} (mode: {mode})")

                    if mode == "chat":
                        messages = [{"role": "user", "content": prompt}]
                        response = client.chat_completion(
                            messages=messages,
                            model=model_name,
                            max_tokens=max_length,
                            temperature=0.7
                        )
                        generated_text = response.choices[0].message.content
                    else:
                        response = client.text_generation(
                            prompt,
                            model=model_name,
                            max_new_tokens=max_length,
                            temperature=0.7,
                            top_p=0.9,
                            return_full_text=False
                        )
                        generated_text = response

                    print(f"✓ Success with {model_name}")
                    return {
                        'response': generated_text.strip(),
                        'model': model_name,
                        'api_status': 'success'
                    }
                except Exception as e:
                    error_str = str(e)
                    if model_name == self.model_name:
                        print(f"✗ Error with {model_name}: {error_str if error_str else 'Empty error - possible auth issue'}")
                        if "503" in error_str or "loading" in error_str.lower():
                            return {
                                'response': "The AI model is currently loading. This takes 20-30 seconds on the first request. Please try again in a moment.",
                                'model': 'api',
                                'api_status': 'model_loading'
                            }
                        if not error_str or "401" in error_str or "403" in error_str:
                            print(f"⚠ Authentication issue detected - check API key")
                        print(f"⚠ Trying alternative free models...")
                    else:
                        error_msg = error_str if error_str else "Empty error"
                        print(f"   Failed: {error_msg[:150]}")
                    continue

            return self._generate_fallback(prompt)

        except ImportError:
            print("✗ huggingface_hub not installed")
            print("  Install with: pip install huggingface_hub")
            return self._generate_fallback(prompt)
        except Exception as e:
            import traceback
            print(f"✗ Error in API generation: {e}")
            print(f"✗ Traceback: {traceback.format_exc()}")
            return self._generate_fallback(prompt)

    def _try_groq(self, prompt: str, max_length: int) -> Optional[Dict]:
        """Try Groq API (free LLaMA)."""
        groq_key = os.environ.get('GROQ_API_KEY')
        if not groq_key or groq_key == 'your-groq-key':
            return None

        try:
            print("⚡ Trying Groq API (free LLaMA)...")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.2-3b-preview",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_length,
                    "temperature": 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                generated_text = data['choices'][0]['message']['content'].strip()
                print(f"✓ Groq success!")
                return {
                    'response': generated_text,
                    'model': 'groq/llama-3.2-3b',
                    'api_status': 'success'
                }
            else:
                print(f"✗ Groq error: {response.status_code}")
                return None

        except Exception as e:
            print(f"✗ Groq error: {e}")
            return None

    def _try_openai(self, prompt: str, max_length: int) -> Optional[Dict]:
        """Try OpenAI API (if key available)."""
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key or openai_key == 'your-openai-key':
            return None

        try:
            print("🤖 Trying OpenAI API...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_length,
                    "temperature": 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                generated_text = data['choices'][0]['message']['content'].strip()
                print(f"✓ OpenAI success!")
                return {
                    'response': generated_text,
                    'model': 'openai/gpt-3.5-turbo',
                    'api_status': 'success'
                }
            else:
                print(f"✗ OpenAI error: {response.status_code}")
                return None

        except Exception as e:
            print(f"✗ OpenAI error: {e}")
            return None

    def _try_ollama(self, prompt: str, max_length: int) -> Optional[Dict]:
        """Try Ollama local API (if available)."""
        try:
            ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
            model = os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')

            print(f"🦙 Trying Ollama at {ollama_url} with model {model}")

            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_length
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('response', '').strip()
                print(f"✓ Ollama success! Generated {len(generated_text)} chars")
                return {
                    'response': generated_text,
                    'model': f'ollama/{model}',
                    'api_status': 'success'
                }
            else:
                print(f"✗ Ollama returned status {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            print("ℹ Ollama not available (install from https://ollama.com)")
            return None
        except Exception as e:
            print(f"✗ Ollama error: {e}")
            return None

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
