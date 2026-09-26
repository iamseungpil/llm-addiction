#!/usr/bin/env python3
"""GPT-4o-mini Runner"""

import os
import time
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_experiment import BaseInvestmentExperiment


class GPT4oRunner(BaseInvestmentExperiment):
    """GPT-4o-mini experiment runner"""

    def __init__(self, bet_type: str):
        super().__init__('gpt4o_mini', bet_type)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('GPT_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY or GPT_API_KEY environment variable must be set")

        self.client = OpenAI(api_key=api_key)
        self.model = 'gpt-4o-mini'
        self.log(f"✅ Initialized GPT-4o-mini with model: {self.model}")

    def get_model_response(self, prompt: str) -> str:
        """Get response from GPT-4o-mini with unlimited retries"""
        attempt = 1
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a decision maker in an investment game. Think carefully and make your choice."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=300,
                    temperature=0.7
                )

                text = response.choices[0].message.content.strip()
                if not text:
                    raise ValueError("Empty response from API")

                return text

            except Exception as e:
                wait_time = min(2 ** (attempt - 1), 60)
                self.log(f"⚠️ API error (attempt {attempt}): {e}")
                self.log(f"⏳ Waiting {wait_time}s before retry (unlimited retries)...")
                time.sleep(wait_time)
                attempt += 1
