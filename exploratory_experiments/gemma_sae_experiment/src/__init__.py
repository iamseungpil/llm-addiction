"""
Gemma SAE Experiment - Version 1: Mechanistic Interpretability
=============================================================

A unified pipeline for SAE-based analysis of gambling behavior in Gemma 2 9B Base model.

Phases:
    1. Feature Extraction - Extract SAE features from decision points
    2. Correlation Analysis - Compare bankrupt vs safe features
    3. Steering Vector - Compute CAA steering vectors
    4. SAE Interpretation - Project steering vectors to SAE space
    5. Steering Experiment - Validate causal effects
    6. SAE Boost - Train residual SAE to reduce reconstruction error
"""

__version__ = "1.0.0"
__author__ = "LLM Addiction Research"
