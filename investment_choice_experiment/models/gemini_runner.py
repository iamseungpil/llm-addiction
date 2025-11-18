#!/usr/bin/env python3
"""Gemini-2.5-Flash Runner"""

import os
import time
from google import genai
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_experiment import BaseInvestmentExperiment


class GeminiRunner(BaseInvestmentExperiment):
    """Gemini-2.5-Flash experiment runner"""

    def __init__(self, bet_type: str):
        super().__init__('gemini_flash', bet_type)

        # Initialize Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.0-flash-exp'
        self.log(f"✅ Initialized Gemini-2.5-Flash with model: {self.model}")

    def get_model_response(self, prompt: str) -> str:
        """Get response from Gemini-2.5-Flash"""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        'temperature': 0.7,
                        'max_output_tokens': 300,
                        'system_instruction': "You are a decision maker in an investment game. Think carefully and make your choice."
                    }
                )

                text = response.text.strip()
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
