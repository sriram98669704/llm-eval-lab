"""
prompts.py — Thin loader. All prompts live in prompts.json.

To add or edit prompts: edit prompts.json only. No code change needed.
Schema: {"category": [{"prompt": "..."}, ...]}
"""

import json
import pathlib


def load_prompts():
    path = pathlib.Path(__file__).parent / "prompts.json"
    with open(path) as f:
        return json.load(f)


PROMPTS = load_prompts()
