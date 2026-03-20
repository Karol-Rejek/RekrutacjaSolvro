import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR   = Path("results")
CHECKPOINTS   = Path("checkpoints")
OUTPUT_DIR    = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

MODES = ["frozen", "full_finetune"]
LABELS = {"frozen": "Frozen backbone", "full_finetune": "Full fine-tuning"}

# ── 1. Krzywe strat ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, mode in enumerate(MODES):
    hist_path = CHECKPOINTS / f"history_{mode}.json"
    if not hist_path.exists():
        continue
    hist = json.loads(hist_path.read_text())
    axes[i].plot(hist["train_loss"], label="Train loss")
    axes[i].plot(hist["val_proxy"],  label="Val proxy (1-F1)")
    axes[i].set_title(LABELS[mode])
    axes[i].set_xlabel("Epoch")
    axes[i].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=150)
plt.close()

# ── 2. Porównanie metryk bar chart ───────────────────────────────────
metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
data = {mode: {} for mode in MODES}
for mode in MODES:
    p = RESULTS_DIR / f"{mode}_metrics.json"
    if p.exists():
        data[mode] = json.loads(p.read_text())

x = np.arange(len(metric_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
for j, mode in enumerate(MODES):
    vals = [data[mode].get(m, 0) for m in metric_names]
    ax.bar(x + j * width, vals, width, label=LABELS[mode])
ax.set_xticks(x + width / 2)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Frozen vs Full fine-tuning — metryki na zbiorze testowym")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=150)
plt.close()

print("Wykresy zapisano w:", OUTPUT_DIR)
