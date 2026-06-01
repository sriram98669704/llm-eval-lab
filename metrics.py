"""
metrics.py — All scoring, extraction, and math helpers (no I/O, no side effects).

Groups of functions:
  - Stability     : cosine_similarity, stability_score, instability_score
  - Speed         : latency_score
  - GSM8K         : extract_model_answer, extract_ground_truth, gsm8k_accuracy
  - HumanEval     : extract_function_code, pass_at_k, humaneval_pass_at_k
  - Task scoring  : category_score (task-aware 0–1 per category),
                    compute_final_score (weighted blend of category scores)

Pure functions — easy to unit test in isolation.
"""

import math
import re

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# -----------------------------
# STABILITY SCORE
# -----------------------------
def stability_score(embeddings):
    """
    Measures consistency across multiple runs via pairwise cosine similarity.
    Higher = more stable/deterministic.
    Returns 1.0 for single runs (trivially consistent).
    """
    if len(embeddings) < 2:
        return 1.0

    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))

    if not sims:
        return 1.0

    return sum(sims) / len(sims)


# -----------------------------
# INSTABILITY SCORE
# -----------------------------
def instability_score(embeddings):
    """
    Measures variation across runs.
    Higher = more inconsistent outputs.
    """
    if len(embeddings) < 2:
        return 0.0

    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))

    if not sims:
        return 0.0

    avg = sum(sims) / len(sims)
    variance = sum((s - avg) ** 2 for s in sims) / len(sims)
    return math.sqrt(variance)


# -----------------------------
# LATENCY SCORE
# -----------------------------
def latency_score(avg_latency):
    """
    Normalize latency to a 0–1 score.
    Lower latency = higher score.
    """
    return 1.0 / (1.0 + avg_latency)


# -----------------------------
# GSM8K ANSWER EXTRACTION
# -----------------------------
def extract_model_answer(text):
    """
    Extract the final numeric answer from a model's GSM8K response.

    Tries 5 patterns from most to least reliable, returning on the first hit.
    Early patterns are the format we explicitly asked for; later ones are
    best-effort fallbacks so a slightly-off response still yields a number.
    """

    # Pattern 1: FINAL ANSWER tag (injected by us)
    match = re.search(r'FINAL ANSWER\s*:\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))

    # Pattern 2: GSM8K native #### format
    match = re.search(r'####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Pattern 3: explicit answer phrases
    match = re.search(
        r'(?:answer is|total is|earned?|result is|paid?|costs?|makes?|writes?|weighs?)\s*:?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if match:
        return float(match.group(1).replace(',', ''))

    # Pattern 4: last standalone number on its own line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        match = re.fullmatch(r'\$?\s*(-?\d[\d,]*(?:\.\d+)?)', line)
        if match:
            return float(match.group(1).replace(',', ''))

    # Pattern 5: fallback — last number in text
    numbers = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
    return float(numbers[-1].replace(',', '')) if numbers else None


def extract_ground_truth(reference):
    """Extract number after #### in GSM8K reference."""
    match = re.search(r'####\s*(-?\d[\d,]*(?:\.\d+)?)', reference)
    return float(match.group(1).replace(',', '')) if match else None


# -----------------------------
# GSM8K ACCURACY
# -----------------------------
def gsm8k_accuracy(gsm8k_results):
    """
    gsm8k_results: list of {"response": str, "reference": str}
    Returns: {"correct": int, "total": int, "accuracy": float}
    """
    correct = 0
    total = len(gsm8k_results)

    for item in gsm8k_results:
        predicted = extract_model_answer(item["response"])
        ground_truth = extract_ground_truth(item["reference"])

        if predicted is not None and ground_truth is not None:
            # GSM8K answers are always integers
            if round(predicted) == round(ground_truth):
                correct += 1

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0
    }


# -----------------------------
# HUMANEVAL CODE EXTRACTION
# -----------------------------
def extract_function_code(response, original_prompt):
    """
    Extract clean Python function from model response.

    Handles:
    - Markdown code blocks (```python ... ```)
    - Model repeating the function signature
    - Model returning just the body
    - Model returning prose + code
    - Missing/duplicate imports
    """
    code = response

    # Step 1: Prefer the FIRST fenced ```python block. Grabbing only the fenced
    # content discards any prose the model wrote before/after the code, which
    # would otherwise become a SyntaxError when executed.
    md_match = re.search(r'```python\s*\n(.*?)```', code, re.DOTALL)
    if md_match:
        code = md_match.group(1)  # keep indentation intact — do NOT strip
    else:
        # No proper block — just remove any stray backtick fences and keep the rest
        code = re.sub(r'```python\s*\n?', '', code)
        code = re.sub(r'```\s*\n?', '', code)

    # Step 2: Extract imports from original prompt (lines before first def)
    import_lines = []
    for line in original_prompt.split('\n'):
        if re.match(r'^def\s+', line):
            break
        if line.strip():
            import_lines.append(line)
    prompt_imports = set(import_lines)
    imports = '\n'.join(import_lines)

    # Step 3: Model returned a full function (the common case — includes 'def').
    if re.search(r'^def\s+', code, re.MULTILINE):
        lines = code.split('\n')
        def_idx = 0
        for i, line in enumerate(lines):
            if re.match(r'^def\s+', line):
                def_idx = i      # first line of the actual function
                break

        # The model may add its own imports (e.g. 'from math import modf') above
        # the def. Keep those, but skip any already present in the prompt to
        # avoid duplicate-import lines.
        model_imports = [
            l for l in lines[:def_idx]
            if re.match(r'^(import|from)\s+', l.strip()) and l.strip() not in prompt_imports
        ]
        func_code = '\n'.join(lines[def_idx:])

        # Merge: original prompt imports + unique model imports + function
        all_imports = []
        if imports:
            all_imports.append(imports)
        if model_imports:
            all_imports.append('\n'.join(model_imports))

        if all_imports:
            return '\n'.join(all_imports) + '\n\n' + func_code
        return func_code

    # Step 4: No 'def' found -> the model returned only the function BODY.
    # Two sub-cases:
    #   (a) Body lines are already indented (4 spaces / tab) — graft directly.
    #   (b) Body lines are NOT indented (llama3 often does this) — indent them first.
    lines = code.split('\n')

    # Find the first non-empty line to determine if body is indented
    code_start = 0
    for i, line in enumerate(lines):
        if line.strip():
            code_start = i
            break

    body_lines = lines[code_start:]

    # Check if ANY non-empty line starts without indentation — if so, indent all
    needs_indent = any(
        line and not line.startswith('    ') and not line.startswith('\t')
        for line in body_lines if line.strip()
    )
    if needs_indent:
        body_lines = [('    ' + line if line.strip() else line) for line in body_lines]

    body = '\n'.join(body_lines)

    # Original prompt already ends with the signature/docstring; append the body
    return original_prompt.rstrip() + '\n' + body


# -----------------------------
# PASS@K (HumanEval paper formula)
# -----------------------------
def pass_at_k(n, c, k):
    """
    Unbiased estimator of pass@k from the HumanEval paper.
    n = total attempts, c = passing attempts, k = k value

    Intuition: probability that at least one of k randomly chosen attempts passes.
    Computed as 1 - P(all k chosen attempts are failures).
    """
    if n < k:
        return 0.0          # can't choose k attempts if we have fewer than k
    if n - c < k:
        return 1.0          # fewer than k failures exist -> every k-subset has a pass
    # 1 - (ways to pick k all-failing) / (ways to pick any k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def humaneval_pass_at_k(humaneval_results, k=1):
    """
    humaneval_results: list of {"passed": bool, "error": str|None}
    Returns pass@1 and pass@k stats.
    When k=1, only pass@1 is returned (no duplicate key).
    When k>1, both pass@1 and pass@k are returned.
    """
    n = len(humaneval_results)
    c = sum(1 for r in humaneval_results if r["passed"])

    result = {
        "pass@1": pass_at_k(n, c, 1),
        "passed": c,
        "total": n
    }

    # Only add pass@k separately when k > 1 to avoid duplicate key
    if k > 1:
        result[f"pass@{k}"] = pass_at_k(n, c, k)

    return result


# -----------------------------
# CATEGORY SCORE (task-aware)
# -----------------------------
def category_score(category, judge_scores=None, gsm8k_results=None, humaneval_results=None, stability=1.0):
    """
    Returns a single 0–1 score for a category using task-appropriate weights.

    Ground truth categories (gsm8k, humaneval) use objective metrics.
    Judge categories use task-specific weighted combinations of judge dimensions.
    """

    # Ground truth — objective, no judge involved
    if category == "gsm8k":
        if not gsm8k_results:
            return 0.0
        return gsm8k_accuracy(gsm8k_results)["accuracy"]

    if category == "humaneval":
        if not humaneval_results:
            return 0.0
        return humaneval_pass_at_k(humaneval_results, k=1)["pass@1"]

    # Judge categories — task-specific dimension weights
    if not judge_scores:
        return 0.0

    # Shorthand for the four judge dimensions (each already 0–1)
    c  = judge_scores.get("correctness", 0.0)
    r  = judge_scores.get("reasoning", 0.0)
    cl = judge_scores.get("clarity", 0.0)
    h  = judge_scores.get("hallucination_free", 0.0)

    # Each profile weights the dimensions that matter most for that task type.
    # Every profile's weights sum to 1.0, so a perfect response scores exactly 1.0.
    PROFILES = {
        "factual":      0.5*c  + 0.3*h  + 0.2*r,    # facts: correct & non-hallucinated
        "math":         0.6*c  + 0.3*r  + 0.1*cl,   # math: right answer & sound steps
        "coding":       0.6*c  + 0.3*r  + 0.1*cl,   # code: correctness dominates
        "creative":     0.5*cl + 0.3*r  + 0.2*c,    # creative: clarity/style first
        "opinion":      0.5*r  + 0.3*cl + 0.2*c,    # opinion: reasoning matters most
        "architecture": 0.4*c  + 0.4*r  + 0.2*stability,  # design: correct + consistent
        "edge_case":    0.6*r  + 0.3*c  + 0.1*cl,   # tricky prompts: reasoning first
    }

    # Unknown categories fall back to a balanced correctness/reasoning blend
    score = PROFILES.get(category, 0.4*c + 0.4*r + 0.2*cl)
    return round(min(max(score, 0.0), 1.0), 4)  # clamp to [0,1] for safety


# -----------------------------
# FINAL SCORE (task-aware)
# -----------------------------
def compute_final_score(category_scores, model_stability, avg_latency):
    """
    Final score built from per-category scores.

    Each category_score is already task-appropriately weighted.
    Final score = 85% avg category performance + 10% stability + 5% speed.

    Handles missing categories gracefully.
    """
    # No task scores at all -> fall back to only the meta signals we do have
    if not category_scores:
        return round(
            0.67 * model_stability + 0.33 * latency_score(avg_latency),
            4
        )

    # Average the per-category scores, then blend in consistency and speed.
    # Task performance dominates (85%) — stability/speed are tie-breakers.
    avg_cat = sum(category_scores.values()) / len(category_scores)

    score = (
        0.85 * avg_cat +
        0.10 * model_stability +
        0.05 * latency_score(avg_latency)
    )

    return round(min(score, 1.0), 4)  # cap at 1.0
