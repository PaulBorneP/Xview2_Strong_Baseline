import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

submission_dict = "submission"

plot_dir = f"plots/{submission_dict}/"
Path(plot_dir).mkdir(parents=True, exist_ok=True)
full_model_metrics_json = "predictions/submission_original/metrics.json"
metrics_file = f"predictions/{submission_dict}/combined_metrics.csv"

with open(full_model_metrics_json, "r") as f:
    full_model_metrics = json.load(f)

full_model_metrics["model_name"] = "full_model"
full_model_metrics["ensemble"] = True

df = pd.read_csv(metrics_file)
df = df.sort_values("model_name")


def add_ensemble_name(row):
    if row["ensemble"]:
        row["model_name"] = row["model_name"] + "_ensemble"
    return row


df = df.apply(add_ensemble_name, axis=1)

for metric in df.columns.drop(["model_name", "ensemble"]):
    plot = df.plot(x="model_name", y=metric, kind="bar")
    plot.axhline(y=full_model_metrics[metric], color="black", linestyle="--")

    plot.set_ylim(0.5, 1)
    plot.set_title("xview2 performance on test set, bar is full ensemble model")
    plt.tight_layout()

    plot_path = str(Path(plot_dir) / f"{metric}.png")
    plt.savefig(plot_path)
