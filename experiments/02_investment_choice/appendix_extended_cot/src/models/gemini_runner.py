#!/usr/bin/env python3
"""Gemini-2.5-Flash Runner for Extended Investment Choice Experiment"""

import os
import time
from google import genai
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_experiment import BaseInvestmentExperiment


class GeminiRunner(BaseInvestmentExperiment):
    """Gemini-2.5-Flash experiment runner"""

    def __init__(self, bet_constraint, bet_type: str):
        super().__init__('gemini_flash', bet_constraint, bet_type)

        # Initialize Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash'
        self.log(f"✅ Initialized Gemini-2.5-Flash with model: {self.model}")

    def get_model_response(self, prompt: str) -> str:
        """Get response from Gemini-2.5-Flash with retry logic"""
        attempt = 1
        max_tokens = 8000

        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        'temperature': 0.7,
                        'max_output_tokens': max_tokens,
                        'system_instruction': f"You are a decision maker in an investment game. Make your choice clearly and respond in under {max_tokens} characters."
                    }
                )

                if response is None:
                    raise ValueError("Response is None")

                if not hasattr(response, 'text') or response.text is None:
                    raise ValueError("Response has no text attribute or text is None")

                text = response.text.strip()
                if not text:
                    raise ValueError("Empty response text from API")

                return text

            except Exception as e:
                wait_time = min(2 ** min(attempt - 1, 6), 60)
                self.log(f"⚠️ API error (attempt {attempt}): {e}")
                self.log(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                attempt += 1
