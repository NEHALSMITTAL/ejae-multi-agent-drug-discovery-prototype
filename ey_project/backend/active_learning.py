# ey_project/backend/active_learning.py
import json
from pathlib import Path
from typing import List, Dict, Any

# Always anchor to PROJECT ROOT (where streamlit_app.py exists)
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # go up to project root
DATA_DIR = PROJECT_ROOT / "ey_project" / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"


def save_feedback(hypothesis: str, accepted: bool) -> bool:
    """
    Save feedback into ey_project/data/feedback.jsonl
    """
    try:
        entry = {"hypothesis": hypothesis, "accepted": bool(accepted)}

        with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[active_learning] WROTE: {entry}")
        print(f"[active_learning] PATH: {FEEDBACK_FILE}")

        return True

    except Exception as e:
        print(f"[active_learning] ERROR: {e}")
        return False


def load_feedback() -> List[Dict[str, Any]]:
    """
    Load feedback file safely.
    """
    if not FEEDBACK_FILE.exists():
        return []

    entries = []
    try:
        with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"[active_learning] ERROR reading: {e}")

    return entries
