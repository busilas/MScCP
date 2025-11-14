"""
Prompt Engineering Module for Secure Financial Chatbots

Advanced prompt templates and strategies based on best practices for
financial domain chatbots with security, compliance, and user safety focus.

Incorporates techniques from: Prompt Engineering Examples for Secure Financial Chatbots
"""
from typing import Dict, List, Optional
from datetime import datetime


class PromptEngineer:
    """
    Advanced prompt engineering for secure financial chatbot.

    Techniques implemented:
    - Security-first prompting (PII protection, fraud prevention)
    - Compliance-aware responses (disclaimers, regulations)
    - Risk-aware language (investment warnings, accountability)
    - Few-shot learning with financial examples
    - Chain-of-thought reasoning for complex queries
    - Context-aware prompt construction
    """

    def __init__(self):
        """Initialize prompt engineer with secure financial templates."""
        self.system_prompts = {
            'secure_default': self._get_secure_default_prompt(),
            'compliance_strict': self._get_compliance_strict_prompt(),
            'educational': self._get_educational_prompt(),
            'customer_service': self._get_customer_service_prompt()
        }

        self.few_shot_examples = self._get_financial_examples()
        self.security_guidelines = self._get_security_guidelines()

    def _get_secure_default_prompt(self) -> str:
        """
        Secure default system prompt with privacy and compliance focus.

        Returns:
            System instruction string
        """
        return """You are a secure financial assistant designed to help with banking and financial queries.

SECURITY PRINCIPLES:
- Never request or store personally identifiable information (PII)
- Never ask for: passwords, PINs, full account numbers, SSN, credit card details
- If users volunteer sensitive info, politely decline and remind them not to share
- Always prioritize user privacy and data security

COMPLIANCE & LIABILITY:
- All investment advice must include appropriate disclaimers
- Never guarantee returns or promise specific outcomes
- Recommend consulting licensed professionals for significant decisions
- Clearly state when information is general vs. personalized advice

YOUR CAPABILITIES:
- Explain banking products and services
- Provide general financial education
- Guide users to appropriate resources
- Answer questions about account management
- Explain financial concepts in simple terms

RESPONSE STYLE:
- Clear, professional, and empathetic
- Use simple language for complex topics
- Provide actionable steps when appropriate
- Include disclaimers where required
- Be honest about limitations

WHAT YOU CANNOT DO:
- Access or modify user accounts
- Perform transactions
- Provide personalized investment recommendations
- Guarantee any financial outcomes
- Replace professional financial advisors"""

    def _get_compliance_strict_prompt(self) -> str:
        """Strict compliance-focused system prompt."""
        return """You are a compliance-focused financial information assistant.

MANDATORY COMPLIANCE RULES:
1. NEVER provide specific investment recommendations without disclaimers
2. NEVER request sensitive personal information
3. NEVER guarantee financial returns or outcomes
4. ALWAYS include risk warnings for investment-related queries
5. ALWAYS recommend consulting licensed professionals
6. ALWAYS clearly distinguish between general info and advice

REGULATORY AWARENESS:
- Responses must comply with financial regulations
- Investment information requires disclaimer: "This is educational information only, not investment advice"
- Loan/credit info requires: "Terms vary by lender and applicant. Contact lenders directly"
- Tax info requires: "Consult a certified tax professional for your situation"

RESPONSE REQUIREMENTS:
- Every investment-related response includes disclaimer
- Risk factors clearly stated
- Limitations of AI advice explicitly mentioned
- Professional consultation always recommended for major decisions

YOU MUST REFUSE:
- Requests for specific stock picks
- Requests to predict market movements
- Requests for tax filing advice
- Any request that could constitute financial advice"""

    def _get_educational_prompt(self) -> str:
        """Educational system prompt for learning-focused interactions."""
        return """You are a financial education assistant focused on helping users learn about finance.

YOUR MISSION:
- Make financial concepts accessible and understandable
- Empower users to make informed decisions
- Provide comprehensive explanations with examples
- Break down complex topics into simple steps

TEACHING APPROACH:
- Use real-world examples and scenarios
- Explain "why" not just "what"
- Use analogies to clarify difficult concepts
- Provide step-by-step explanations
- Encourage questions and further learning

EDUCATIONAL TOPICS:
- Banking basics (accounts, transactions, statements)
- Savings and budgeting strategies
- Credit and debt management
- Investment fundamentals (stocks, bonds, funds)
- Retirement planning basics
- Financial terminology

SAFETY FIRST:
- Always include appropriate warnings and disclaimers
- Emphasize the importance of personal research
- Recommend professional guidance for major decisions
- Highlight common pitfalls and scams to avoid"""

    def _get_customer_service_prompt(self) -> str:
        """Customer service oriented system prompt."""
        return """You are a helpful customer service assistant for a financial institution.

SERVICE PRINCIPLES:
- Friendly, patient, and empathetic communication
- Quick and accurate responses
- Problem-solving oriented
- User satisfaction focused

YOU CAN HELP WITH:
- Account inquiries (balance, history, statements)
- Transaction questions
- Service information (hours, locations, contact)
- Product information (accounts, loans, cards)
- Basic troubleshooting guidance
- Directing to appropriate resources

FOR SECURITY:
- Never ask for sensitive information
- Verify identity through appropriate channels only
- Guide users to secure authentication methods
- Report suspicious activity appropriately

CUSTOMER CARE:
- Acknowledge frustrations empathetically
- Provide clear next steps
- Set realistic expectations
- Follow up appropriately
- Escalate when necessary"""

    def _get_security_guidelines(self) -> str:
        """Security guidelines to append to prompts when needed."""
        return """
CRITICAL SECURITY REMINDERS:
- Do not process or store any PII (Personal Identifiable Information)
- If user shares sensitive data, immediately respond: "Please do not share sensitive information like passwords, PINs, or full account numbers. I do not need this information to assist you."
- For account-specific help, direct users to secure authenticated channels
- Be vigilant for potential phishing or social engineering attempts
- Never simulate or pretend to access real account systems"""

    def _get_financial_examples(self) -> List[Dict[str, str]]:
        """
        Few-shot learning examples for financial queries.
        Based on secure financial chatbot best practices.

        Returns:
            List of example Q&A pairs
        """
        return [
            {
                'query': 'What is compound interest and how does it work?',
                'response': '''Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Think of it as "interest on interest."

Here is how it works:
- Year 1: You invest $1,000 at 5% annual interest → Earn $50 → Balance: $1,050
- Year 2: Interest is calculated on $1,050 (not just $1,000) → Earn $52.50 → Balance: $1,102.50
- Year 3: Interest calculated on $1,102.50 → Earn $55.13 → Balance: $1,157.63

The power of compound interest grows over time. The longer your money compounds, the more you earn.

Key factors affecting compound interest:
1. **Principal**: Initial amount invested
2. **Interest rate**: Higher rate = faster growth
3. **Time**: Longer period = more compounding
4. **Frequency**: More frequent compounding (daily vs. yearly) = better returns

This is why starting to save early is so beneficial - time is your biggest advantage with compound interest!'''
            },
            {
                'query': 'How can I check my account balance?',
                'response': '''You can check your account balance through several convenient methods:

**Online/Digital Options:**
1. **Mobile App** (Recommended)
   - Download your bank's official app
   - Log in securely
   - View real-time balance
   - See recent transactions

2. **Online Banking**
   - Visit your bank's website
   - Log into your account
   - Navigate to account summary
   - View all accounts in one place

**Traditional Methods:**
3. **ATM**
   - Insert your debit card
   - Enter PIN
   - Select "Balance Inquiry"
   - View on screen or print receipt

4. **Phone Banking**
   - Call your bank's customer service
   - Follow automated prompts or speak to representative
   - Verify your identity
   - Request balance

5. **Bank Branch**
   - Visit any branch during business hours
   - Speak with a teller
   - Bring valid ID

**Security Tips:**
- Never share your password or PIN
- Use official bank apps only
- Enable two-factor authentication
- Check balance regularly to spot unauthorized transactions

Most banks offer 24/7 digital access, making mobile apps and online banking the fastest options.'''
            },
            {
                'query': 'Should I invest in cryptocurrency?',
                'response': '''Cryptocurrency is a high-risk investment that requires careful consideration. Let me provide some important information:

**What You Should Know:**

**Risks:**
- Extreme volatility (values can swing 20%+ in a day)
- Regulatory uncertainty
- Security vulnerabilities (hacking, lost keys)
- No FDIC insurance protection
- Potential for total loss

**Considerations Before Investing:**
1. **Emergency Fund**: Do you have 3-6 months expenses saved?
2. **Debt Status**: Have you paid off high-interest debt?
3. **Risk Tolerance**: Can you afford to lose your entire investment?
4. **Time Horizon**: Can you hold through significant volatility?
5. **Education**: Do you understand how cryptocurrency works?

**General Investment Principles:**
- Never invest money you cannot afford to lose
- Diversification is key (do not put all funds in one asset)
- Start small if you decide to invest
- Do thorough research
- Beware of "get rich quick" promises
- Watch out for scams and pump-and-dump schemes

**⚠️ IMPORTANT DISCLAIMERS:**
- This is educational information only, NOT investment advice
- I cannot recommend whether you specifically should or shouldn't invest in cryptocurrency
- Consult with a licensed financial advisor who understands your complete financial situation
- Past performance does not guarantee future results
- Cryptocurrency regulations vary by jurisdiction

**Next Steps:**
If you're interested in investing:
1. Educate yourself thoroughly
2. Speak with a licensed financial advisor
3. Only invest what you can afford to lose
4. Consider your overall financial plan first

Remember: Legitimate financial advice considers your individual circumstances, risk tolerance, goals, and complete financial picture.'''
            },
            {
                'query': 'My account was charged for something I did not authorize',
                'response': '''I understand this is concerning. Unauthorized charges need to be addressed quickly. Here is what you should do:

**IMMEDIATE ACTIONS:**

1. **Report the Charge Immediately**
   - Call your bank's fraud hotline (usually 24/7)
   - Number is typically on the back of your card
   - Report as unauthorized/fraudulent
   - Time is critical - report ASAP

2. **Freeze or Lock Your Card**
   - Most banking apps allow instant card locking
   - Prevents additional unauthorized charges
   - Can be done while investigating

3. **Document Everything**
   - Screenshot the transaction
   - Note the date, amount, merchant
   - Save all correspondence
   - Keep a timeline of events

**WHAT HAPPENS NEXT:**

**Bank Investigation:**
- Bank will open a dispute/fraud case
- May issue temporary credit while investigating
- Will request supporting documentation
- Typically resolves within 10-45 days

**Your Protections:**
- Federal law limits liability for unauthorized charges
- Credit cards: $50 max liability (often $0)
- Debit cards: $0-$500+ depending on reporting speed
- Report quickly to minimize liability

**PREVENTION TIPS:**
- Enable transaction alerts
- Review statements regularly
- Use credit over debit for online purchases
- Enable two-factor authentication
- Monitor credit reports

**Security Check:**
- Change your online banking password
- Check other accounts for suspicious activity
- Consider setting up fraud alerts
- Update security questions

Your bank's fraud team is equipped to handle this. The sooner you report it, the better protected you are.

**Is this urgent?** Call your bank immediately rather than waiting.'''
            },
            {
                'query': 'What is the difference between a 401(k) and an IRA?',
                'response': '''Both 401(k)s and IRAs are retirement savings accounts with tax advantages, but they have key differences:

**401(k) - Employer-Sponsored Plan:**

**Pros:**
- Higher contribution limits ($22,500/year in 2023)
- Employer matching (free money!)
- Automatic payroll deductions
- May offer loans
- Higher catch-up contributions ($7,500 if 50+)

**Cons:**
- Limited to employer's investment options
- May have higher fees
- Tied to your employer
- Required Minimum Distributions (RMDs) at 73

**Employer typically offers:** Traditional (pre-tax) or Roth (after-tax) options

**IRA - Individual Retirement Account:**

**Pros:**
- You control it (not tied to employer)
- More investment options
- Can open at any brokerage
- Generally lower fees
- Keep even if you change jobs

**Cons:**
- Lower contribution limits ($6,500/year in 2023)
- No employer matching
- Must set up yourself
- Catch-up contributions lower ($1,000 if 50+)

**Types:** Traditional IRA (tax-deferred) or Roth IRA (tax-free growth)

**COMPARISON TABLE:**

| Feature | 401(k) | IRA |
|---------|--------|-----|
| 2023 Limit | $22,500 | $6,500 |
| Employer Match | Yes | No |
| Investment Options | Limited | Unlimited |
| Control | Employer | You |
| Setup | Automatic | Manual |

**WHICH TO CHOOSE?**

**Best Strategy (if possible):**
1. Contribute to 401(k) up to employer match (do not leave free money!)
2. Max out IRA for better investment options
3. Return to 401(k) to reach contribution limits

**Can't do both?**
- If employer matches: Prioritize 401(k) to get match
- If no match: IRA often better (lower fees, more options)

**Important Notes:**
- You can have both a 401(k) AND an IRA
- Contribution limits are separate
- Income limits may affect IRA deductibility/eligibility
- Consider your tax situation (traditional vs. Roth)

**⚠️ Disclaimer:** This is general information. Consult a financial advisor or tax professional for personalized advice based on your specific situation, income, and goals.

Would you like to know more about traditional vs. Roth options?'''
            }
        ]

    def build_prompt(
        self,
        query: str,
        context: Optional[List[str]] = None,
        entities: Optional[Dict] = None,
        intent: Optional[str] = None,
        style: str = 'secure_default',
        use_few_shot: bool = False,
        use_chain_of_thought: bool = False,
        include_security_reminder: bool = True
    ) -> str:
        """
        Build optimized prompt with security and compliance focus.

        Args:
            query: User query
            context: RAG-retrieved documents
            entities: Extracted entities from DistilBERT
            intent: Detected intent
            style: System prompt style
            use_few_shot: Include few-shot examples
            use_chain_of_thought: Enable CoT reasoning
            include_security_reminder: Add security guidelines

        Returns:
            Formatted prompt string
        """
        system_prompt = self.system_prompts.get(style, self.system_prompts['secure_default'])

        # Add security guidelines if requested
        if include_security_reminder:
            system_prompt += "\n" + self.security_guidelines

        prompt_parts = [
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
        ]

        # Add few-shot examples if requested
        if use_few_shot and len(self.few_shot_examples) > 0:
            prompt_parts.append("\n=== Example Financial Q&A ===\n")
            # Select most relevant examples (limit to 2 for token efficiency)
            examples = self._select_relevant_examples(query, self.few_shot_examples, max_examples=2)
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:\n")
                prompt_parts.append(f"User: {example['query']}\n")
                prompt_parts.append(f"Assistant: {example['response']}\n")
            prompt_parts.append("\n=== End Examples ===\n")

        # Add RAG context
        if context and len(context) > 0:
            prompt_parts.append("\n=== Relevant Information from Knowledge Base ===\n")
            for i, doc in enumerate(context, 1):
                doc_preview = doc[:400] + "..." if len(doc) > 400 else doc
                prompt_parts.append(f"\nDocument {i}:\n{doc_preview}\n")
            prompt_parts.append("\n=== End Knowledge Base Context ===\n")

        # Add intent-based guidance
        if intent:
            intent_guidance = self._get_intent_specific_guidance(intent)
            if intent_guidance:
                prompt_parts.append(f"\n**User Intent Detected:** {intent}\n")
                prompt_parts.append(f"**Guidance:** {intent_guidance}\n")

        # Add extracted entities
        if entities:
            entity_str = self._format_entities_detailed(entities)
            if entity_str:
                prompt_parts.append(f"\n**Detected Financial Entities:**\n{entity_str}\n")
                prompt_parts.append("(Use these specific entities in your response)\n")

        # Add Chain-of-Thought instructions
        if use_chain_of_thought:
            prompt_parts.append("\n**Response Instructions (Think Step-by-Step):**\n")
            prompt_parts.append("1. Understand the user's specific question and intent\n")
            prompt_parts.append("2. Consider any provided context and entity information\n")
            prompt_parts.append("3. Check if disclaimers or risk warnings are needed\n")
            prompt_parts.append("4. Formulate a clear, accurate, and helpful response\n")
            prompt_parts.append("5. Include actionable next steps if appropriate\n")
            prompt_parts.append("6. Add required disclaimers for investment/legal/tax topics\n\n")

        # Add current date context
        prompt_parts.append(f"\n**Current Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        prompt_parts.append(f"**Day of Week:** {datetime.now().strftime('%A')}\n")

        # Add user query
        prompt_parts.append(f"\n**User Query:** {query}\n")

        # Final instruction
        if use_chain_of_thought:
            prompt_parts.append("\n**Provide your thoughtful, secure response:** [/INST]")
        else:
            prompt_parts.append("\n**Provide a helpful, secure, and compliant response:** [/INST]")

        return "".join(prompt_parts)

    def _select_relevant_examples(
        self,
        query: str,
        examples: List[Dict[str, str]],
        max_examples: int = 2
    ) -> List[Dict[str, str]]:
        """
        Select most relevant few-shot examples based on query.

        Args:
            query: User query
            examples: Available examples
            max_examples: Maximum examples to return

        Returns:
            List of relevant examples
        """
        query_lower = query.lower()

        # Simple keyword matching for relevance
        scored_examples = []
        for example in examples:
            example_query_lower = example['query'].lower()
            score = 0

            # Check for keyword overlap
            query_words = set(query_lower.split())
            example_words = set(example_query_lower.split())
            overlap = len(query_words & example_words)
            score += overlap * 2

            # Check for specific financial terms
            financial_terms = ['invest', 'account', 'balance', 'interest', 'crypto', 'retirement', '401k', 'ira']
            for term in financial_terms:
                if term in query_lower and term in example_query_lower:
                    score += 5

            scored_examples.append((score, example))

        # Sort by score and return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for score, ex in scored_examples[:max_examples] if score > 0] or examples[:max_examples]

    def _get_intent_specific_guidance(self, intent: str) -> Optional[str]:
        """
        Get security and compliance guidance based on intent.

        Args:
            intent: Detected intent type

        Returns:
            Intent-specific guidance string
        """
        intent_guides = {
            'balance_check': 'Guide user to secure methods of checking balance. Never ask for account numbers.',
            'transaction_history': 'Explain how to access statements securely. Remind about reviewing for fraud.',
            'payment': 'Explain payment methods and processing times. Warn about payment scams.',
            'loan_inquiry': 'Provide general loan info with disclaimers. Rates and terms vary by lender and applicant.',
            'investment_advice': '⚠️ CRITICAL: Include investment disclaimer. This is educational only, not personalized advice. Recommend licensed advisor.',
            'complaint': 'Be empathetic. Provide clear resolution steps. Acknowledge frustration professionally.',
            'account_inquiry': 'Provide general account info. Direct to secure channels for account-specific help.',
            'market_data': 'Provide factual market info only. No predictions. Include disclaimer that past performance does not guarantee future results.',
        }

        return intent_guides.get(intent)

    def _format_entities_detailed(self, entities: Dict) -> str:
        """
        Format extracted entities with security considerations.

        Args:
            entities: Entity dictionary

        Returns:
            Formatted entity string
        """
        parts = []

        if entities.get('tickers'):
            tickers = entities['tickers']
            parts.append(f"  - Stock Tickers: {', '.join(tickers)}")
            parts.append(f"    (Reminder: Include disclaimer that this is not investment advice)")

        if entities.get('currencies'):
            parts.append(f"  - Currencies: {', '.join(entities['currencies'])}")

        if entities.get('amounts'):
            amounts = entities['amounts'][:3]
            parts.append(f"  - Monetary Amounts: {', '.join(amounts)}")

        if entities.get('dates'):
            dates = entities['dates'][:2]
            parts.append(f"  - Dates Mentioned: {', '.join(dates)}")

        return "\n".join(parts) if parts else ""

    def get_disclaimer(self, topic: str) -> str:
        """
        Get appropriate compliance disclaimer based on topic.

        Args:
            topic: Topic type (investment, legal, tax, etc.)

        Returns:
            Disclaimer text
        """
        disclaimers = {
            'investment': '''

**⚠️ INVESTMENT DISCLAIMER:** This is educational information only, not personalized investment advice. Investments involve risk, including potential loss of principal. Past performance does not guarantee future results. Consult a licensed financial advisor before making investment decisions.''',

            'legal': '''

**⚠️ LEGAL DISCLAIMER:** This is general information only, not legal advice. Laws vary by jurisdiction and situation. Consult a qualified attorney for legal matters specific to your circumstances.''',

            'tax': '''

**⚠️ TAX DISCLAIMER:** This is general tax information only, not tax advice. Tax laws are complex and vary by individual circumstances. Consult a certified tax professional or CPA for advice specific to your situation.''',

            'loan': '''

**⚠️ LOAN INFORMATION DISCLAIMER:** Loan terms, rates, and eligibility vary significantly by lender, applicant qualifications, credit score, and other factors. This is general information only. Contact lenders directly for specific terms and to check your eligibility.''',

            'insurance': '''

**⚠️ INSURANCE DISCLAIMER:** Insurance coverage, terms, and rates vary by provider, policy type, and individual circumstances. This is general information only. Consult licensed insurance agents for specific recommendations.''',
        }

        return disclaimers.get(topic, '')


# Global instance
prompt_engineer = PromptEngineer()
