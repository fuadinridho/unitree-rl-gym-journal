"""
Legged Gym Scripts Module

This module provides utility scripts for:
- Training RL policies (train.py)
- Playing/Exporting policies (play.py)
- Analyzing policies (policy_analyzer.py)
- Processing analysis data (data_processing_guide.py)

Usage:
    # Run scripts directly
    python -m legged_gym.scripts.train --task=g1
    python -m legged_gym.scripts.play --task=g1
    python -m legged_gym.scripts.policy_analyzer
    python -m legged_gym.scripts.data_processing_guide

    # Or import as module
    from legged_gym.scripts.policy_analyzer import PolicyAnalyzer
    from legged_gym.scripts.data_processing_guide import load_analysis_data
"""

from pathlib import Path

# Get the scripts directory
SCRIPTS_DIR = Path(__file__).parent

__all__ = [
    'train',
    'play',
    'policy_analyzer',
    'data_processing_guide',
    'SCRIPTS_DIR',
]

__version__ = '1.0.0'
__author__ = 'Unitree RL Gym'
