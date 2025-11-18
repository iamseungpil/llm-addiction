#!/usr/bin/env python3
"""GPT-4.1-mini Runner (using gpt-4o-mini-2024-07-18)"""

import os
import time
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_experiment import BaseInvestmentExperiment


class GPT41Runner(BaseInvestmentExperiment):
    """GPT-4.1-mini experiment runner"""

    def __init__(self, bet_type: str):
        super().__init__('gpt41_mini', bet_type)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('GPT_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY or GPT_API_KEY environment variable must be set")

        self.client = OpenAI(api_key=api_key)
        # Using the specific model version for GPT-4.1-mini
        self.model = 'gpt-4o-mini-2024-07-18'
        self.log(f"✅ Initialized GPT-4.1-mini with model: {self.model}")

    def get_model_response(self, prompt: str) -> str:
        """Get response from GPT-4.1-mini"""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
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
                self.log(f"⚠️ API error (attempt {attempt}/{max_retries}): {e}")

                if attempt == max_retries:
                    self.log(f"❌ Failed after {max_retries} attempts, using fallback (Option 1)")
                    return "I choose Option 1"  # Conservative fallback: Stop

                self.log(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
