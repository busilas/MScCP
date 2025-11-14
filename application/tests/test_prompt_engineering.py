"""
Tests for Prompt Engineering Module
"""
import pytest
from app.services.prompt_engineering import prompt_engineer


class TestPromptEngineer:
    """Test suite for PromptEngineer class."""

    def test_initialization(self):
        """Test that prompt engineer initializes correctly."""
        assert prompt_engineer is not None
        assert len(prompt_engineer.system_prompts) == 4
        assert len(prompt_engineer.few_shot_examples) == 5

    def test_system_prompts_exist(self):
        """Test that all system prompt styles are available."""
        expected_styles = ['secure_default', 'compliance_strict', 'educational', 'customer_service']
        for style in expected_styles:
            assert style in prompt_engineer.system_prompts
            assert len(prompt_engineer.system_prompts[style]) > 0

    def test_few_shot_examples(self):
        """Test that few-shot examples are properly formatted."""
        examples = prompt_engineer.few_shot_examples

        assert len(examples) == 5

        for example in examples:
            assert 'query' in example
            assert 'response' in example
            assert len(example['query']) > 0
            assert len(example['response']) > 0

    def test_build_basic_prompt(self):
        """Test basic prompt building."""
        prompt = prompt_engineer.build_prompt(
            query="What is my account balance?",
            style='secure_default'
        )

        assert '<s>[INST]' in prompt
        assert '[/INST]' in prompt
        assert 'What is my account balance?' in prompt
        assert 'secure financial assistant' in prompt.lower()

    def test_build_prompt_with_context(self):
        """Test prompt building with RAG context."""
        context = [
            "Account balance can be checked via mobile app",
            "Online banking provides real-time balance"
        ]

        prompt = prompt_engineer.build_prompt(
            query="How do I check my balance?",
            context=context,
            style='secure_default'
        )

        assert 'Relevant Information from Knowledge Base' in prompt
        assert 'mobile app' in prompt
        assert 'online banking' in prompt.lower()

    def test_build_prompt_with_entities(self):
        """Test prompt building with extracted entities."""
        entities = {
            'tickers': ['AAPL', 'GOOGL'],
            'amounts': ['$5000'],
            'currencies': ['USD']
        }

        prompt = prompt_engineer.build_prompt(
            query="Should I invest $5000 in AAPL?",
            entities=entities,
            style='secure_default'
        )

        assert 'Detected Financial Entities' in prompt
        assert 'AAPL' in prompt
        assert '$5000' in prompt

    def test_build_prompt_with_intent(self):
        """Test prompt building with detected intent."""
        prompt = prompt_engineer.build_prompt(
            query="Should I invest in stocks?",
            intent='investment_advice',
            style='secure_default'
        )

        assert 'User Intent Detected' in prompt
        assert 'investment_advice' in prompt
        assert 'disclaimer' in prompt.lower()

    def test_few_shot_learning(self):
        """Test prompt with few-shot examples."""
        prompt = prompt_engineer.build_prompt(
            query="What is compound interest?",
            use_few_shot=True,
            style='secure_default'
        )

        assert 'Example Financial Q&A' in prompt
        assert 'Example 1:' in prompt

    def test_chain_of_thought(self):
        """Test prompt with chain-of-thought reasoning."""
        prompt = prompt_engineer.build_prompt(
            query="Should I invest in crypto?",
            use_chain_of_thought=True,
            style='secure_default'
        )

        assert 'Response Instructions' in prompt
        assert 'Step-by-Step' in prompt or 'step-by-step' in prompt.lower()

    def test_security_guidelines_included(self):
        """Test that security guidelines are included."""
        prompt = prompt_engineer.build_prompt(
            query="What's my password?",
            include_security_reminder=True,
            style='secure_default'
        )

        assert 'SECURITY' in prompt
        assert 'PII' in prompt or 'sensitive information' in prompt.lower()

    def test_compliance_strict_style(self):
        """Test compliance strict style includes mandatory disclaimers."""
        prompt = prompt_engineer.build_prompt(
            query="Should I buy stocks?",
            style='compliance_strict'
        )

        assert 'compliance' in prompt.lower()
        assert 'disclaimer' in prompt.lower()
        assert 'NEVER' in prompt or 'never' in prompt.lower()

    def test_educational_style(self):
        """Test educational style for learning-focused responses."""
        prompt = prompt_engineer.build_prompt(
            query="Explain savings accounts",
            style='educational'
        )

        assert 'educational' in prompt.lower() or 'education' in prompt.lower()
        assert 'learn' in prompt.lower() or 'teach' in prompt.lower()

    def test_customer_service_style(self):
        """Test customer service style."""
        prompt = prompt_engineer.build_prompt(
            query="I need help with my account",
            style='customer_service'
        )

        assert 'customer service' in prompt.lower() or 'service' in prompt.lower()
        assert 'help' in prompt.lower() or 'assist' in prompt.lower()

    def test_get_disclaimer_investment(self):
        """Test investment disclaimer."""
        disclaimer = prompt_engineer.get_disclaimer('investment')

        assert 'DISCLAIMER' in disclaimer
        assert 'investment' in disclaimer.lower()
        assert 'not' in disclaimer.lower() and 'advice' in disclaimer.lower()

    def test_get_disclaimer_tax(self):
        """Test tax disclaimer."""
        disclaimer = prompt_engineer.get_disclaimer('tax')

        assert 'DISCLAIMER' in disclaimer
        assert 'tax' in disclaimer.lower()
        assert 'professional' in disclaimer.lower()

    def test_get_disclaimer_legal(self):
        """Test legal disclaimer."""
        disclaimer = prompt_engineer.get_disclaimer('legal')

        assert 'DISCLAIMER' in disclaimer
        assert 'legal' in disclaimer.lower()
        assert 'attorney' in disclaimer.lower()

    def test_intent_specific_guidance(self):
        """Test that intent-specific guidance is provided."""
        guidance = prompt_engineer._get_intent_specific_guidance('investment_advice')

        assert guidance is not None
        assert 'disclaimer' in guidance.lower()

    def test_select_relevant_examples(self):
        """Test example selection based on query relevance."""
        examples = prompt_engineer._select_relevant_examples(
            query="What is compound interest?",
            examples=prompt_engineer.few_shot_examples,
            max_examples=2
        )

        assert len(examples) <= 2
        assert len(examples) > 0

    def test_format_entities_detailed(self):
        """Test detailed entity formatting."""
        entities = {
            'tickers': ['AAPL', 'TSLA'],
            'amounts': ['$1000', '$500'],
            'currencies': ['USD', 'EUR']
        }

        formatted = prompt_engineer._format_entities_detailed(entities)

        assert 'Stock Tickers' in formatted
        assert 'AAPL' in formatted
        assert 'Monetary Amounts' in formatted
        assert '$1000' in formatted

    def test_security_reminder_present(self):
        """Test that security reminders are present in prompts."""
        prompt = prompt_engineer.build_prompt(
            query="Test query",
            include_security_reminder=True
        )

        assert 'SECURITY' in prompt or 'security' in prompt.lower()
        assert 'PII' in prompt

    def test_date_context_included(self):
        """Test that current date is included in prompts."""
        prompt = prompt_engineer.build_prompt(query="Test")

        assert 'Current Date' in prompt

    def test_full_integration_prompt(self):
        """Test full prompt with all features enabled."""
        prompt = prompt_engineer.build_prompt(
            query="Should I invest $10000 in Bitcoin?",
            context=["Bitcoin is volatile", "Cryptocurrency risks"],
            entities={'currencies': ['Bitcoin'], 'amounts': ['$10000']},
            intent='investment_advice',
            style='compliance_strict',
            use_few_shot=True,
            use_chain_of_thought=True,
            include_security_reminder=True
        )

        # Check all components are present
        assert '<s>[INST]' in prompt
        assert 'compliance' in prompt.lower()
        assert 'Example' in prompt
        assert 'Step-by-Step' in prompt or 'step' in prompt.lower()
        assert 'Bitcoin' in prompt
        assert '$10000' in prompt
        assert 'investment_advice' in prompt
        assert 'Relevant Information' in prompt
        assert 'SECURITY' in prompt or 'security' in prompt.lower()
        assert '[/INST]' in prompt


class TestPromptEngineeringIntegration:
    """Test integration with other services."""

    def test_import_from_llama_service(self):
        """Test that prompt_engineer can be imported in llama_service."""
        try:
            from app.services.llama_service import LLaMAService
            llama = LLaMAService()
            # If import succeeded, test passes
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import: {e}")

    def test_prompt_engineer_singleton(self):
        """Test that prompt_engineer is a singleton."""
        from app.services.prompt_engineering import prompt_engineer as pe1
        from app.services.prompt_engineering import prompt_engineer as pe2

        assert pe1 is pe2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
