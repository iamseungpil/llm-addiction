#!/usr/bin/env python3
"""
Check OpenAI API rate limit status
"""

import openai
import json
from datetime import datetime

# Test API call to check rate limit
try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    print("✅ API working - Rate limit reset!")
    print(f"Response: {response.choices[0].message.content}")
    
except openai.RateLimitError as e:
    print("❌ Still rate limited")
    print(f"Error: {e}")
    
    # Extract wait time from error message
    error_str = str(e)
    if "try again in" in error_str:
        import re
        wait_time = re.search(r'try again in ([\d.]+)s', error_str)
        if wait_time:
            seconds = float(wait_time.group(1))
            print(f"Wait time: {seconds} seconds ({seconds/3600:.1f} hours)")
    
except Exception as e:
    print(f"Other error: {e}")

print(f"\nCurrent time: {datetime.now()}")
print("Rate limit resets at UTC midnight (24 hours from first request)")