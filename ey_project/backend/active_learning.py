import json
from pathlib import Path

# ------------------------------
# Directory + File Setup
# ------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)  # Ensure the folder exists

FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"


# ------------------------------
# Save Feedback
# ------------------------------

def save_feedback(hypothesis: str, accepted: bool):
    """
    Saves feedback about a hypothesis to data/feedback.jsonl
    """
    entry = {
        "hypothesis": hypothesis,
        "accepted": accepted
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------
# Load Feedback
# ------------------------------

def load_feedback():
    """
    Loads all feedback (if file exists).
    Returns a list of dicts.
    """
    if not FEEDBACK_FILE.exists():
        return []

    feedback_entries = []

    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                feedback_entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    return feedback_entries
