"""
Save trained ensemble + metadata to models/research/ with versioned name.

Usage (programmatically):
    from scripts.save_experiment_artifact import save_artifact
    save_artifact(
        experiment_id="A1_pct3",
        result=trained_ensemble_result,
        horizon=5,
        target_definition="fwd_5d_ret >= 0.03",
        train_size=176496,
        test_size=4000,
        oos_auc=0.6556,
        feature_importances={"top_15": [...], "groups": {...}},
        notes="A1 threshold sweep, +3pct absolute return target",
    )
"""
import json
import joblib
from datetime import datetime
from pathlib import Path

RESEARCH_DIR = Path("models/research")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def save_artifact(
    experiment_id,
    result,
    horizon,
    target_definition,
    train_size,
    test_size,
    oos_auc,
    feature_importances,
    notes="",
):
    """Save model + metadata. Returns (model_path, meta_path)."""
    today = datetime.now().strftime("%Y%m%d")
    base = f"{experiment_id}_h{horizon}d_{today}"
    model_path = RESEARCH_DIR / f"{base}.joblib"
    meta_path = RESEARCH_DIR / f"{base}.meta.json"

    joblib.dump(result, model_path)

    meta = {
        "experiment_id": experiment_id,
        "saved_at": datetime.now().isoformat(),
        "horizon": horizon,
        "target_definition": target_definition,
        "train_size": train_size,
        "test_size": test_size,
        "oos_auc": float(oos_auc),
        "feature_importances": feature_importances,
        "notes": notes,
        "feature_cols": list(getattr(result, "feature_cols", [])),
        "model_filename": model_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return model_path, meta_path


def list_artifacts():
    """Return all experiment metadata as a list."""
    out = []
    for meta_file in sorted(RESEARCH_DIR.glob("*.meta.json")):
        try:
            out.append(json.loads(meta_file.read_text()))
        except Exception:
            continue
    return out


if __name__ == "__main__":
    print(f"Experiments in {RESEARCH_DIR}:")
    for m in list_artifacts():
        print(f"  {m['experiment_id']}: AUC={m['oos_auc']:.4f}  ({m['saved_at'][:10]})")
