"""
LLM Interface Module for Chain-of-Alpha MVP

Provides a unified interface for interacting with different LLM providers
"""

import logging
from typing import Dict, List, Any, Optional
import re
import json
import time

logger = logging.getLogger(__name__)

class LLMInterface:
    """
    Unified interface for LLM interactions
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('llm_model', 'llama-3-8b')
        
        # Try to get API key from config or environment
        import os
        self.api_key = (
            config.get('llm_api_key') or 
            os.getenv('GROK_API_KEY') or 
            os.getenv('OPENAI_API_KEY') or
            os.getenv('LLM_API_KEY')
        )
        
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)

        # Initialize the appropriate LLM client
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on model type"""
        if self.model_name.startswith('llama') or self.model_name == 'llama':
            return LocalLlamaClient(self.config)
        elif self.model_name == 'grok':
            return GrokClient(self.config)
        elif self.model_name == 'openai':
            return OpenAIClient(self.config)
        else:
            logger.warning(f"Unknown model {self.model_name}, defaulting to mock client")
            return MockLLMClient(self.config)

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response from the LLM

        Args:
            prompt: The prompt to send to the LLM
            context: Optional context data

        Returns:
            Generated response string
        """
        try:
            logger.debug(f"Sending prompt to {self.model_name}")
            response = self.client.generate_response(prompt, context)

            # Clean and validate response
            response = self._clean_response(response)

            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_fallback_response(prompt)

    def _clean_response(self, response: str) -> str:
        """Clean and validate LLM response"""
        if not response or len(response.strip()) == 0:
            return "No response generated"

        # Remove excessive whitespace
        response = re.sub(r'\n+', '\n', response.strip())

        # Limit response length
        if len(response) > self.max_tokens * 4:  # Rough character limit
            response = response[:self.max_tokens * 4] + "..."

        return response

    def _get_fallback_response(self, prompt: str) -> str:
        """Provide fallback response when LLM fails"""
        logger.warning("Using fallback response due to LLM failure")

        if "factor" in prompt.lower():
            return """
Here are some example alpha factors:

1. df['momentum_20'] * df['volume_ratio']
2. df['rsi'] / df['volatility_20']
3. (df['close'] - df['sma_20']) / df['sma_20']
4. df['macd'] * df['returns']
5. df['volume_ma_5'] / df['volume_ma_20']
"""
        else:
            return "Default response: Please check your configuration and try again."

    def validate_factor_expression(self, expression: str) -> bool:
        """
        Validate if a factor expression is syntactically correct Python
        """
        try:
            # Basic syntax check
            compile(expression, '<string>', 'eval')
            return True
        except SyntaxError:
            logger.warning(f"Invalid factor expression: {expression}")
            return False

class BaseLLMClient:
    """Base class for LLM clients"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response - to be implemented by subclasses"""
        raise NotImplementedError

class LocalLlamaClient(BaseLLMClient):
    """Client for local Llama models via Hugging Face"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the Llama model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            import os
            from dotenv import load_dotenv

            # Load environment variables
            load_dotenv()

            # Use the approved Llama model that you now have access to
            model_name = "meta-llama/Llama-3.2-3B-Instruct"

            logger.info(f"Loading {model_name}...")

            # Check for Hugging Face token
            hf_token = os.getenv('HUGGINGFACE_TOKEN') or self.config.get('huggingface_token')
            
            if not hf_token:
                logger.error("Hugging Face token not found. Please set HUGGINGFACE_TOKEN environment variable or add 'huggingface_token' to config.")
                logger.info("Get your token from: https://huggingface.co/settings/tokens")
                logger.info("You also need to request access to the Llama model at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
                raise ValueError("Hugging Face authentication required")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=hf_token,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Llama model loaded successfully")

        except ImportError:
            logger.error("Transformers library not available. Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            logger.info("Falling back to mock client")
            raise

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using local Llama model"""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            import torch

            # Format prompt
            full_prompt = self._format_prompt(prompt, context)

            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + self.config.get('max_tokens', 500),
                    temperature=self.config.get('temperature', 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(full_prompt):].strip()

            return response

        except Exception as e:
            logger.error(f"Llama generation failed: {e}")
            raise

    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt for Llama"""
        system_prompt = """You are an expert quantitative analyst specializing in alpha factor generation for algorithmic trading. You help create innovative, interpretable factors that can predict stock returns.

CRITICAL: All factor expressions must use pandas DataFrame syntax with df['column_name']. Never use bare column names.

Available data columns include: df['open'], df['high'], df['low'], df['close'], df['volume'], df['returns'], df['log_returns'], df['volume_ma_5'], df['volume_ma_20'], df['volume_ratio'], df['momentum_5'], df['momentum_20'], df['volatility_20'], df['sma_5'], df['sma_20'], df['sma_50'], df['rsi'], df['macd'], df['macd_signal'], df['macd_hist']

Generate factors as Python expressions that can be evaluated on pandas DataFrames. Always use df['column'] syntax.

Example: df['close'] / df['sma_20'] - 1  (NOT: close / sma_20 - 1)"""

        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        else:
            context_str = ""

        return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}{context_str}\n<|assistant|>"

class GrokClient(BaseLLMClient):
    """Client for Grok API with structured factor generation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.factor_schema = {
            "type": "object",
            "properties": {
                "factor_expression": {"type": "string", "description": "Valid pandas expression using df['column'] syntax"},
                "explanation": {"type": "string", "description": "Economic intuition behind the factor"},
                "expected_signal": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "category": {"type": "string", "enum": ["momentum", "mean_reversion", "volatility", "volume", "cross_sectional", "fundamental"]}
            },
            "required": ["factor_expression", "explanation", "expected_signal", "confidence", "category"]
        }

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Grok API with structured prompting"""
        try:
            import requests
            import json

            import os
            api_key = (
                self.config.get('llm_api_key') or 
                os.getenv('GROK_API_KEY') or
                os.getenv('LLM_API_KEY')
            )
            if not api_key:
                raise ValueError("Grok API key not provided. Set 'llm_api_key' in config or environment variable GROK_API_KEY")

            url = "https://api.x.ai/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Enhanced system prompt for finance-specific reasoning
            system_prompt = """You are a world-class quantitative analyst specializing in systematic alpha factor generation. You have deep expertise in:

1. Market microstructure and behavioral finance
2. Cross-sectional and time-series momentum effects  
3. Mean reversion patterns and volatility clustering
4. Volume-price relationships and market regime identification
5. Risk factor neutralization and portfolio construction

CRITICAL REQUIREMENTS:
- Generate factors using pandas DataFrame syntax with df['column_name']
- Ensure factors are market-neutral (remove beta exposure)
- Focus on non-obvious relationships that capture behavioral biases
- Consider market regimes (trending, ranging, volatile, calm)
- Validate expressions are executable and meaningful

Available columns: {columns}
Current market context: {market_context}

Response Format: Return a JSON object matching this schema:
{schema}"""

            # Build context-aware prompt
            if context and 'data_analysis' in context:
                columns_str = ", ".join(context['data_analysis'].get('columns', []))
                market_context = f"Period: {context['data_analysis'].get('date_range', {}).get('start', 'Unknown')} to {context['data_analysis'].get('date_range', {}).get('end', 'Unknown')}"
            else:
                columns_str = "df['close'], df['open'], df['high'], df['low'], df['volume'], df['returns'], df['log_returns'], df['sma_5'], df['sma_20'], df['rsi'], df['macd']"
                market_context = "Multi-year historical data across various market regimes"

            formatted_system = system_prompt.format(
                columns=columns_str,
                market_context=market_context,
                schema=json.dumps(self.factor_schema, indent=2)
            )

            messages = [
                {
                    "role": "system", 
                    "content": formatted_system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            data = {
                "model": "grok-3",
                "messages": messages,
                "temperature": self.config.get('temperature', 0.7),
                "max_tokens": self.config.get('max_tokens', 1500),
                "response_format": {"type": "json_object"} if "json" in prompt.lower() else None
            }

            logger.info(f"Sending request to Grok API with {len(prompt)} char prompt")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to parse as JSON for structured factors
            try:
                json_content = json.loads(content)
                logger.info("Successfully parsed structured JSON response from Grok")
                return self._format_structured_response(json_content)
            except json.JSONDecodeError:
                logger.warning("Grok response not valid JSON, returning raw content")
                return content

        except Exception as e:
            logger.error(f"Grok API call failed: {e}")
            raise

    def _format_structured_response(self, json_response: dict) -> str:
        """Format structured JSON response back to expected string format"""
        try:
            factor_expr = json_response.get('factor_expression', '')
            explanation = json_response.get('explanation', '')
            confidence = json_response.get('confidence', 0)
            category = json_response.get('category', 'unknown')
            
            formatted = f"""FACTOR: {factor_expr}
EXPLANATION: {explanation}
CONFIDENCE: {confidence:.2f}
CATEGORY: {category}"""
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting structured response: {e}")
            return f"FACTOR: {json_response.get('factor_expression', 'Error parsing factor')}\nEXPLANATION: {json_response.get('explanation', 'Error parsing explanation')}"

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API"""

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using OpenAI API"""
        try:
            import openai

            api_key = self.config.get('llm_api_key')
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            client = openai.OpenAI(api_key=api_key)

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert quantitative analyst specializing in alpha factor generation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 1000)
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

class MockLLMClient(BaseLLMClient):
    """Mock client for testing and fallback"""

    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate mock response"""
        logger.info("Using mock LLM client")

        # Simulate some processing time
        time.sleep(0.5)

        if "factor" in prompt.lower() and "generate" in prompt.lower():
            # Generate a single factor response in the expected format
            # Use only basic columns that we know exist: close, open, high, low, volume, returns, log_returns
            factors = [
                "df['close'] / df['close'].shift(5) - 1",  # 5-day momentum
                "df['returns'].rolling(10).std()",  # 10-day volatility
                "(df['high'] - df['low']) / df['close']",  # Normalized range
                "df['volume'] / df['volume'].shift(5)",  # Volume ratio
                "df['log_returns'].rolling(5).mean()",  # 5-day return mean
                "(df['close'] - df['open']) / df['open']",  # Intraday return
                "df['returns'].rolling(20).skew()",  # Return skewness
                "df['close'] / df['close'].rolling(20).mean() - 1",  # Price vs 20-day MA
                "df['volume'].rolling(10).std() / df['volume'].rolling(10).mean()",  # Volume volatility
                "df['returns'].shift(1) * df['returns']"  # Return momentum
            ]
            
            explanations = [
                "5-day price momentum factor",
                "10-day volatility measure",
                "Normalized intraday price range",
                "5-day volume growth ratio", 
                "5-day average return",
                "Intraday gap return",
                "20-day return distribution skewness",
                "Price deviation from 20-day mean",
                "Volume volatility ratio",
                "Return serial correlation"
            ]
            
            # Return a single factor each time (cycling through them)
            import random
            idx = random.randint(0, len(factors) - 1)
            
            return f"""FACTOR: {factors[idx]}
EXPLANATION: {explanations[idx]}"""
        elif "improve" in prompt.lower() or "optimize" in prompt.lower():
            return """
Based on the backtest results, here are optimization suggestions:

1. Add volatility scaling: factor * (1 / df['volatility_20'])
2. Include volume confirmation: factor * df['volume_ratio']
3. Add trend filter: factor * (df['sma_20'] / df['sma_50'] - 1)
4. Implement decay: factor * df['returns'].ewm(span=10).mean()
5. Add momentum confirmation: factor * df['momentum_5']

The optimized factor should show improved Sharpe ratio and reduced drawdown.
"""
        else:
            return "Mock response: This is a placeholder response for testing purposes."