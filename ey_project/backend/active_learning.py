import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"

# Ensure file exists
if not FEEDBACK_FILE.exists():
    FEEDBACK_FILE.write_text("", encoding="utf-8")

# ---------------------------------------------------------------
# Save Feedback
# ---------------------------------------------------------------
def save_feedback(hypothesis: str, accepted: bool) -> bool:
    """
    Append feedback as a JSONL entry.
    """
    try:
        entry = {"hypothesis": hypothesis, "accepted": bool(accepted)}

        with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()   # IMPORTANT: prevents missing writes

        return True

    except Exception as e:
        print(f"[feedback error] {e}")
        return False


# ---------------------------------------------------------------
# Load Feedback
# ---------------------------------------------------------------
def load_feedback() -> List[Dict[str, Any]]:
    if not FEEDBACK_FILE.exists():
        return []

    entries = []
    with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return entries


# ---------------------------------------------------------------
# Compute Acceptance Score
# ---------------------------------------------------------------
def compute_acceptance_score(feedback_entries: List[Dict[str, Any]]) -> Dict[str, float]:
    if not feedback_entries:
        return {}

    accept_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for fb in feedback_entries:
        hyp = fb["hypothesis"]
        total_counts[hyp] += 1
        if fb["accepted"]:
            accept_counts[hyp] += 1

    scores = {}
    for hyp in total_counts:
        scores[hyp] = accept_counts[hyp] / total_counts[hyp]

    return scores


# ---------------------------------------------------------------
# Re-Rank Hypotheses
# ---------------------------------------------------------------
def rerank_hypotheses(hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Boost hypotheses with positive feedback.
    """
    feedback = load_feedback()
    scores = compute_acceptance_score(feedback)

    for h in hypotheses:
        text = h.get("text")
        h["feedback_score"] = scores.get(text, 0.5)

    return sorted(hypotheses, key=lambda x: x["feedback_score"], reverse=True)
