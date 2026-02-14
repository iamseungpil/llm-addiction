"""Quick test for Gemma model loading fix"""
import sys
sys.path.insert(0, 'exploratory_experiments/alternative_paradigms/src')

from common.model_loader import ModelLoader
from common.utils import setup_logger

logger = setup_logger(__name__)

# Test Gemma loading
logger.info("Testing Gemma model loading...")
model_loader = ModelLoader("gemma", gpu_id=0)
model_loader.load()

# Test generation
test_prompt = "What is 2+2?"
logger.info(f"Test prompt: {test_prompt}")
response = model_loader.generate(test_prompt, max_new_tokens=20)
logger.info(f"Response: {response}")

# Cleanup
model_loader.unload()
logger.info("✅ Test passed!")
