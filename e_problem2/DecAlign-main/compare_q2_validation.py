"""Click Run in VS Code to compare completed or in-progress validation histories."""
from __future__ import annotations

import json
from pathlib import Path

project = Path(__file__).resolve().parent.parent
experiments = (
    ("original: 70% missing, no class weight", "decalign_q2_aligned_final"),
    ("30% missing, no class weight", "decalign_aligned_less_missing"),
    ("30% missing, mild class weight", "decalign_aligned_balanced"),
    ("no artificial missing, flat head", "decalign_aligned_flat_clean"),
    ("no artificial missing, two-stage head", "decalign_aligned_two_stage_clean"),
    ("no artificial missing, two-stage head + matched dropout", "decalign_aligned_two_stage_dropout_clean"),
    ("no artificial missing, two-stage + mild class weight", "decalign_aligned_two_stage_weighted_clean"),
)

print("Compare clean validation Acc/F1/MAE; missing F1 is reported only for augmented runs.")
for label, folder in experiments:
    path = project / "runs" / folder / "history.json"
    if not path.is_file():
        print(f"{label}: not run yet")
        continue
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        print(f"{label}: no epochs recorded")
        continue
    best = max(history, key=lambda item: item["selection_score"])
    clean = best["valid_clean"]
    missing = best["valid_missing_half"]
    missing_f1 = (sum(missing[name]["macro_f1"] for name in ("text", "audio", "vision")) / 3
                  if missing else None)
    run_dir = project / "runs" / folder
    finished = (run_dir / "result.json").is_file() or (run_dir / "validation_complete.json").is_file()
    missing_text = f"{missing_f1:.3f}" if missing_f1 is not None else "not simulated"
    print(f"{label}: best_epoch={best['epoch']} val_acc={clean['accuracy']:.2%} "
          f"val_F1={clean['macro_f1']:.3f} missing_F1={missing_text} "
          f"val_MAE={clean['mae']:.3f} "
          f"{'finished' if finished else 'training may still be running'}")
