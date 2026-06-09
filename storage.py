"""
storage.py — All file I/O for the benchmark.

Responsibilities:
  - init_run()     : create the timestamped run folder, set module-level path
  - log_result()   : append one run record to results.jsonl
  - save_meta()    : write meta.json
  - save_summary() : write summary.json
  - update_index() : append this run's entry to data/index.json

Single source of truth: every artifact is written into the timestamped run
folder under data/runs/<timestamp>/ and is never overwritten. data/index.json
is the registry of all runs; the dashboard reads the newest run from there via
latest_run_dir().

Folder structure after a run:
  data/
  ├── index.json
  └── runs/
      └── 2024-01-15T10-30-00/
          ├── results.jsonl
          ├── summary.json
          ├── routing.json
          └── meta.json
"""

import json
import pathlib
from datetime import datetime

# -----------------------------
# MODULE-LEVEL RUN STATE
# Set once by init_run() — all writes use this path for the duration of the run.
# -----------------------------
_run_dir = None   # pathlib.Path to data/runs/<timestamp>/


# -----------------------------
# INITIALISE RUN FOLDER
# -----------------------------
def init_run(started_at: datetime) -> pathlib.Path:
    """
    Create the timestamped run folder and set the module-level _run_dir.
    Called once at the start of run_benchmark() in main.py.

    Folder name uses dashes instead of colons (colons are invalid on Windows).
    Returns the run folder path so main.py can reference it if needed.
    """
    global _run_dir

    folder_name = started_at.strftime("%Y-%m-%dT%H-%M-%S")
    _run_dir = pathlib.Path("data") / "runs" / folder_name
    _run_dir.mkdir(parents=True, exist_ok=True)

    return _run_dir


def _ensure_init():
    """Guard: raises clearly if storage is used before init_run() is called."""
    if _run_dir is None:
        raise RuntimeError(
            "storage.init_run() must be called before any writes. "
            "Call it at the start of run_benchmark()."
        )


# -----------------------------
# FIND THE LATEST RUN
# -----------------------------
def latest_run_dir():
    """
    Return the path to the most recent run folder, or None if no runs exist.

    Prefers the last entry in data/index.json (the authoritative run registry).
    Falls back to the lexically-newest folder under data/runs/ if the index is
    missing or unreadable — folder names are sortable timestamps, so the last
    one alphabetically is the newest.

    This is how the dashboard and route()/load_routing() locate the current run
    without needing to know its timestamp.
    """
    runs_root = pathlib.Path("data") / "runs"

    # Preferred: the last entry in the index registry.
    index_path = pathlib.Path("data") / "index.json"
    if index_path.exists():
        try:
            runs = json.loads(index_path.read_text()).get("runs", [])
            if runs:
                candidate = runs_root / runs[-1]["id"]
                if candidate.exists():
                    return candidate
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # Fallback: newest folder on disk (timestamp names sort lexically).
    if runs_root.exists():
        subdirs = sorted((d for d in runs_root.iterdir() if d.is_dir()), reverse=True)
        if subdirs:
            return subdirs[0]

    return None


# -----------------------------
# WRITE HELPERS (write into the active run folder)
# -----------------------------
def _write_json(filename, data):
    """Write JSON into the active run folder."""
    _ensure_init()
    (_run_dir / filename).write_text(json.dumps(data, indent=2))


def _append_jsonl(filename, record):
    """Append one JSON line into the active run folder's file."""
    _ensure_init()
    with open(_run_dir / filename, "a") as f:
        f.write(json.dumps(record) + "\n")


# -----------------------------
# LOG ONE RUN RESULT
# -----------------------------
def log_result(
    model,
    category,
    prompt,
    response,
    latency,
    cost_usd=None,
    judge_score=None,
):
    """
    Append one individual run record (model × prompt) to results.jsonl.
    Called after every model call in models.py.
    """
    record = {
        "timestamp":   datetime.now().isoformat(),
        "model":       model,
        "category":    category,
        "prompt":      prompt,
        "response":    response,
        "latency":     latency,
        "cost_usd":    cost_usd,
        "judge_score": judge_score,
    }
    _append_jsonl("results.jsonl", record)


# -----------------------------
# SAVE SUMMARY
# -----------------------------
def save_summary(summary_dict):
    """
    Write summary.json — aggregated scores per model.
    Read by the Streamlit dashboard.
    """
    _write_json("summary.json", summary_dict)


# -----------------------------
# SAVE REPRODUCIBILITY METADATA
# -----------------------------
def save_meta(meta_dict):
    """
    Write meta.json — full reproducibility receipt for this run.
    Includes git SHA, environment, config snapshot, judge health.
    """
    _write_json("meta.json", meta_dict)


# -----------------------------
# SAVE ROUTING TABLE
# -----------------------------
def save_routing(routing_dict):
    """
    Write routing.json — the machine-readable routing policy.
    Schema: { "generated_at": ..., "policy": { category: { default, escalate_to, bar, ... } } }
    Written to both the run folder and root so the dashboard always has the latest.
    """
    _write_json("routing.json", routing_dict)


# -----------------------------
# LOAD ROUTING TABLE
# -----------------------------
def load_routing():
    """
    Read the routing policy from the most recent run's routing.json.

    Returns the full routing dict { "generated_at": ..., "policy": {...} },
    or None if no run exists yet (no benchmark has been run) or it can't be
    parsed. Used by router.route() when no policy is passed in.

    No _ensure_init() guard: reading the latest routing table is independent of
    any active run, so this is safe to call standalone (e.g. from route()).
    """
    run_dir = latest_run_dir()
    if run_dir is None:
        return None
    path = run_dir / "routing.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# -----------------------------
# UPDATE RUN INDEX
# -----------------------------
def update_index(entry: dict):
    """
    Append this run's summary entry to data/index.json.
    Creates the file if it doesn't exist yet.

    index.json is the registry of all runs — the dashboard reads it
    to populate a run history dropdown (Step 18).

    Each entry contains just enough to be useful without duplicating
    all of meta.json:
      id, started_at, finished_at, duration_seconds,
      git_sha, git_dirty, run_fingerprint, models, prompt_counts
    """
    index_path = pathlib.Path("data") / "index.json"

    # Read existing index or start fresh
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except (json.JSONDecodeError, OSError):
            index = {"runs": []}
    else:
        index = {"runs": []}

    index["runs"].append(entry)
    index_path.write_text(json.dumps(index, indent=2))
