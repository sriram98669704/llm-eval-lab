"""
categorizer.py — kNN live-prompt categorization (Step 22).

Maps a free-text prompt to the nearest of the 6 Tilicho task categories using
cosine similarity over text-embedding-3-small vectors.

Two functions:
  build_fingerprints() — embed all 30 known prompts, average 5 per category
                         → 6 fingerprint vectors. Call once at startup.
  categorize(prompt)   — embed the prompt, cosine-similarity against the 6
                         fingerprints, return the nearest category name.
"""

import math

from dispatcher import embed
from config import CATEGORIES
from prompts import PROMPTS  # single loader for prompts.json (path-safe via __file__)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _average_vectors(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    avg = [sum(v[i] for v in vectors) / n for i in range(dim)]
    return avg


def build_fingerprints():
    """
    Embed all 30 known prompts (5 per category) and average them into 6
    fingerprint vectors — one per category.

    Returns a dict: { category_name: [float, ...] }

    Makes 30 embedding API calls sequentially. The OpenAI embeddings API
    accepts a list of strings in one request, so this could be collapsed to
    1 call — a worthwhile optimisation for the deployed app (Step 23) where
    every new browser session pays this cost. Not done yet; correctness first.
    Call once at dashboard startup and cache in st.session_state — do not
    call on every user interaction.
    """
    raw = PROMPTS

    # Collect all 30 prompts in one flat list and send them in a single
    # batched API call — one request instead of 30 sequential ones.
    all_prompts = []
    order = []  # [(category, count)] so we can slice the flat result back
    for category in CATEGORIES:
        cat_prompts = [p["prompt"] for p in raw.get(category, [])]
        all_prompts.extend(cat_prompts)
        order.append((category, len(cat_prompts)))

    all_vectors = embed(all_prompts)  # one API call, returns list of vectors

    fingerprints = {}
    idx = 0
    for category, count in order:
        fingerprints[category] = _average_vectors(all_vectors[idx: idx + count])
        idx += count
    return fingerprints


def categorize(prompt, fingerprints):
    """
    Embed the prompt and return the nearest category by cosine similarity.

    Args:
        prompt       — the free-text prompt to classify
        fingerprints — dict returned by build_fingerprints()

    Returns:
        (category, score) where category is one of the 6 CATEGORIES strings
        and score is the cosine similarity (0–1, higher = more confident).
    """
    vec = embed(prompt)
    best_cat = None
    best_score = -1.0
    for category, fingerprint in fingerprints.items():
        score = _cosine(vec, fingerprint)
        if score > best_score:
            best_score = score
            best_cat = category
    return best_cat, best_score
