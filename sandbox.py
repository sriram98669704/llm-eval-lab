"""
sandbox.py — Isolated code executor for HumanEval.

Writes model-generated code (function + hidden unit tests) to a temp file and
runs it in a separate python3 subprocess with a hard timeout. The subprocess
boundary means buggy or malicious code can't affect the main process.

Catches and reports: syntax errors, runtime errors, assertion (test) failures,
and infinite loops (via timeout). Always cleans up the temp file.

Returns: {"passed": bool, "error": str|None, "time": float}
"""

import subprocess
import tempfile
import os
import time

# -----------------------------
# CODE SANDBOX
# -----------------------------
def run_code_in_sandbox(code, timeout=5):
    """
    Runs Python code in an isolated subprocess with timeout.

    Handles:
    - Syntax errors
    - Runtime errors
    - Infinite loops (timeout)
    - Assertion failures (test failures)

    Returns:
        {"passed": bool, "error": str|None, "time": float}
    """
    tmp_path = None

    try:
        # Write to temp file in /tmp
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir='/tmp'
        ) as f:
            f.write(code)
            tmp_path = f.name

        start = time.perf_counter()

        result = subprocess.run(
            ['python3', tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        elapsed = round(time.perf_counter() - start, 3)

        if result.returncode == 0:
            return {"passed": True, "error": None, "time": elapsed}
        else:
            return {"passed": False, "error": result.stderr.strip(), "time": elapsed}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout after {timeout}s", "time": float(timeout)}

    except Exception as e:
        return {"passed": False, "error": str(e), "time": 0.0}

    finally:
        # Always clean up temp file even if crash
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
