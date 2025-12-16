#!/usr/bin/env python3
"""Claude-3.5-Haiku Runner"""

import os
import time
import anthropic
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_experiment import BaseInvestmentExperiment


class ClaudeRunner(BaseInvestmentExperiment):
    """Claude-3.5-Haiku experiment runner"""

    def __init__(self, bet_type: str):
        super().__init__('claude_haiku', bet_type)

        # Initialize Anthropic client
        api_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable must be set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = 'claude-3-5-haiku-latest'
        self.log(f"✅ Initialized Claude-3.5-Haiku with model: {self.model}")

    def get_model_response(self, prompt: str) -> str:
        """Get response from Claude-3.5-Haiku"""
        attempt = 1
        while True:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    temperature=0.7,
                    system="You are a decision maker in an investment game. Think carefully and make your choice.",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                text = response.content[0].text.strip()
                if not text:
                    raise ValueError("Empty response from API")

                return text

            except Exception as e:
                wait_time = min(2 ** (attempt - 1), 60)
                self.log(f"⚠️ API error (attempt {attempt}): {e}")
                self.log(f"⏳ Waiting {wait_time}s before retry (unlimited retries)...")
                time.sleep(wait_time)
                attempt += 1
