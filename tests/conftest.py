"""
pytest configuration and shared fixtures.
"""

import os

import pytest

# Set a dummy API key so Settings can be instantiated in tests
# without requiring a real .env file.
os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-real")
