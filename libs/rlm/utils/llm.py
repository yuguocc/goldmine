"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key,base_url="https://openrouter.ai/api/v1")

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "") or ""))
                    else:
                        parts.append(str(getattr(item, "text", "") or ""))
                return "".join(parts)
            if content is None:
                content = getattr(message, "reasoning", None) or getattr(
                    message,
                    "refusal",
                    None,
                )
            return "" if content is None else str(content)

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
