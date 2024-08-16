"""
    Adapted code from Raj Magesh
"""
import hashlib
import json


def hash_string(string: str) -> str:
    return hashlib.sha1(string.encode()).hexdigest()[:10]


def hash_configs(configs: dict) -> str:
    return hash_string(json.dumps(configs))
