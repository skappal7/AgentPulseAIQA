"""
LLM Coaching Module
Supports OpenRouter API and Local LLM (LM Studio / Ollama)
"""

import requests
import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str  # 'openrouter' or 'local'
    model: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000


class LLMCoachingEngine:
    """Generate coaching insights using LLM"""
    
    # Free models from OpenRouter
    OPENROUTER_FREE_MODELS = [
        "deepseek/deepseek-chat-v3.1:free",
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "qwen/qwen3-235b-a22b:free",
        "mistralai/mistral-7b-instruct:free",
        "openchat/openchat-7b:free",
        "gryphe/mythomax-l2-13b:free",
        "openai/gpt-oss-20b:free",
        "meta-llama/llama-4-maverick:free",
        "moonshotai/kimi-vl-a3b-thinking:free",
        "moonshotai/kimi-k2:free"
    ]
    
    # Model context limits (approximate)
    MODEL_CONTEXT_LIMITS = {
        "deepseek/deepseek-chat-v3.1:free": 64000,
        "deepseek/deepseek-r1-distill-llama-70b:free": 32000,
        "meta-llama/llama-3.3-70b-instruct:free": 8000,
        "qwen/qwen2.5-vl-32b-instruct:free": 32000,
        "qwen/qwen3-235b-a22b:free": 32000,
        "mistralai/mistral-7b-instruct:free": 8000,
        "openchat/openchat-7b:free": 8000,
        "gryphe/mythomax-l2-13b:free": 8000,
        "openai/gpt-oss-20b:free": 8000,
        "meta-llama/llama-4-maverick:free": 16000,
        "moonshotai/kimi-vl-a3b-thinking:free": 32000,
        "moonshotai/kimi-k2:free": 16000,
    }
    
    def __init__(self, config: LLMConfig, cache_enabled: bool = True):
        self.config = config
        self.cache_enabled = cache_enabled
        self.cache = {}
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)"""
        return len(text) // 4
    
    def _truncate_to_context(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within context window"""
        estimated_tokens = self._estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Keep proportional amount
        ratio = max_tokens / estimated_tokens
        target_chars = int(len(text) * ratio * 0.9)  # 90% to be safe
        
        return text[:target_chars] + "\n\n[... truncated for context limit ...]"
    
    def get_model_context_limit(self, model: str) -> int:
        """Get context limit for model"""
        return self.MODEL_CONTEXT_LIMITS.get(model, 8000)
    
    def _call_openrouter(self, prompt: str) -> Dict[str, Any]:
        """Call OpenRouter API"""
        if not self.config.api_key:
            raise ValueError("OpenRouter API key not provided")
        
        # Get context limit for model
        context_limit = self.get_model_context_limit(self.config.model)
        safe_limit = int(context_limit * 0.7)  # Reserve 30% for response
        
        # Truncate prompt if needed
        truncated_prompt = self._truncate_to_context(prompt, safe_limit)
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://agentpulse.ai",
            "X-Title": "AgentPulse AI"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": truncated_prompt
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'content': data['choices'][0]['message']['content'],
                'model': self.config.model,
                'usage': data.get('usage', {}),
                'truncated': len(truncated_prompt) < len(prompt)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'content': None
            }
    
    def _call_local_llm(self, prompt: str) -> Dict[str, Any]:
        """Call local LLM (LM Studio / Ollama)"""
        if not self.config.endpoint:
            raise ValueError("Local LLM endpoint not provided")
        
        # Assume context limit of 4096 for local models
        safe_limit = 3000
        truncated_prompt = self._truncate_to_context(prompt, safe_limit)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": truncated_prompt
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            response = requests.post(
                self.config.endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats (LM Studio vs Ollama)
            if 'choices' in data:
                content = data['choices'][0]['message']['content']
            elif 'response' in data:
                content = data['response']
            else:
                content = str(data)
            
            return {
                'success': True,
                'content': content,
                'model': self.config.model,
                'usage': data.get('usage', {}),
                'truncated': len(truncated_prompt) < len(prompt)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'content': None
            }
    
    def generate_coaching(
        self,
        agent_name: str,
        transcripts: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        categories: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Generate coaching insights for an agent
        
        Args:
            agent_name: Name of the agent
            transcripts: List of classified transcript dicts
            metrics: Dict with AHT, NPS, sentiment, etc.
            categories: Category distribution dict
            
        Returns:
            Dict with coaching insights
        """
        
        # Build prompt
        prompt = self._build_coaching_prompt(agent_name, transcripts, metrics, categories)
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(prompt)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Call LLM based on provider
        if self.config.provider == 'openrouter':
            result = self._call_openrouter(prompt)
        elif self.config.provider == 'local':
            result = self._call_local_llm(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
        
        if not result['success']:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'agent_name': agent_name
            }
        
        # Parse response
        coaching_data = self._parse_coaching_response(result['content'], agent_name, metrics, categories)
        coaching_data['model_used'] = result['model']
        coaching_data['usage'] = result.get('usage', {})
        coaching_data['truncated'] = result.get('truncated', False)
        
        # Cache result
        if self.cache_enabled:
            self.cache[cache_key] = coaching_data
        
        return coaching_data
    
    def _build_coaching_prompt(
        self,
        agent_name: str,
        transcripts: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        categories: Dict[str, int]
    ) -> str:
        """Build coaching prompt from agent data"""
        
        # Sample up to 5 transcripts
        sample_transcripts = transcripts[:5]
        
        transcript_text = "\n\n".join([
            f"Transcript {i+1}:\nCategory: {t.get('category', 'N/A')}\nSubcategory: {t.get('subcategory', 'N/A')}\nText: {t.get('redacted_transcript', '')[:500]}..."
            for i, t in enumerate(sample_transcripts)
        ])
        
        categories_text = "\n".join([f"- {cat}: {count}" for cat, count in categories.items()])
        
        metrics_text = "\n".join([f"- {key}: {value}" for key, value in metrics.items()])
        
        prompt = f"""You are an expert contact center QA coach. Analyze this agent's performance and provide actionable coaching insights.

AGENT: {agent_name}

PERFORMANCE METRICS:
{metrics_text}

CATEGORY DISTRIBUTION:
{categories_text}

SAMPLE CONVERSATIONS (Redacted):
{transcript_text}

Please provide coaching insights in the following JSON format:

{{
  "root_cause": "Primary issue or pattern identified",
  "coaching_points": [
    "Specific coaching point 1",
    "Specific coaching point 2",
    "Specific coaching point 3"
  ],
  "sample_script": "Example of improved response for common scenario",
  "kpi_recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ],
  "strengths": [
    "Positive aspect 1",
    "Positive aspect 2"
  ],
  "priority": "High/Medium/Low"
}}

Focus on:
1. Communication clarity and empathy
2. Policy compliance
3. Efficiency improvements
4. Customer satisfaction impact

Provide actionable, specific recommendations."""

        return prompt
    
    def _parse_coaching_response(
        self,
        response: str,
        agent_name: str,
        metrics: Dict[str, Any],
        categories: Dict[str, int]
    ) -> Dict[str, Any]:
        """Parse LLM response into structured coaching data"""
        
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                coaching = json.loads(json_str)
            else:
                # Fallback: create structured response from text
                coaching = {
                    'root_cause': 'See coaching points below',
                    'coaching_points': response.split('\n')[:5],
                    'sample_script': 'N/A',
                    'kpi_recommendations': [],
                    'strengths': [],
                    'priority': 'Medium'
                }
        except json.JSONDecodeError:
            # Fallback parsing
            coaching = {
                'root_cause': 'Analysis provided below',
                'coaching_points': [line.strip() for line in response.split('\n') if line.strip()],
                'sample_script': 'See coaching points for guidance',
                'kpi_recommendations': [],
                'strengths': [],
                'priority': 'Medium'
            }
        
        return {
            'success': True,
            'agent_name': agent_name,
            'coaching': coaching,
            'metrics': metrics,
            'categories': categories,
            'raw_response': response
        }
    
    def batch_generate_coaching(
        self,
        agents_data: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Generate coaching for multiple agents
        
        Args:
            agents_data: List of agent data dicts
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of coaching results
        """
        results = []
        total = len(agents_data)
        
        for i, agent_data in enumerate(agents_data):
            if progress_callback:
                progress_callback(i + 1, total)
            
            result = self.generate_coaching(
                agent_name=agent_data['agent_name'],
                transcripts=agent_data['transcripts'],
                metrics=agent_data.get('metrics', {}),
                categories=agent_data.get('categories', {})
            )
            
            results.append(result)
        
        return results
