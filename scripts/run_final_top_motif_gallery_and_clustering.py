from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports" / "final_visual_evidence" / "scripts"))

from final_visual_evidence_core import main


if __name__ == "__main__":
    main(["gallery"])
