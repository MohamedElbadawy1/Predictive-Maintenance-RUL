"""
frontend/api_client.py — thin HTTP client for the Predictive
Maintenance RUL API. Kept separate from app.py (the Streamlit page)
so it can be tested with plain `requests` calls against a real server,
without needing a Streamlit runtime.
"""
from typing import Dict, List

import requests

DEFAULT_TIMEOUT = 30


def check_health(api_url: str) -> Dict:
    response = requests.get(f"{api_url}/health", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def check_readiness(api_url: str) -> Dict:
    response = requests.get(f"{api_url}/health/ready", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def predict(api_url: str, readings: List[Dict]) -> Dict:
    response = requests.post(f"{api_url}/predict/", json={"readings": readings}, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()
