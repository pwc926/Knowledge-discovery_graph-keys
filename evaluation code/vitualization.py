import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [
    {"Method": "SAKey baseline",  "Precision": 0.002, "Recall": 1.000, "F1": 0.004, "Pred": 167319},
    {"Method": "SAKey optimized", "Precision": 0.974, "Recall": 0.639, "F1": 0.772, "Pred": 196},
    {"Method": "GraphKey",        "Precision": 0.976, "Recall": 0.405, "F1": 0.572, "Pred": 124},
]

df = pd.DataFrame(data)

# Precision Recall F1 
metrics = ["Precision", "Recall", "F1"]
x = np.arange(len(df["Method"]))
width = 0.25

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

bars = []

for j, m in enumerate(metrics):
    b = ax.bar(x + (j-1)*width, df[m], width, label=m)
    bars.append(b)

ax.set_xticks(x)
ax.set_xticklabels(df["Method"], rotation=15)
ax.set_ylim(0,1.05)
ax.set_ylabel("Score")
ax.set_title("SAKey baseline vs SAKey optimized vs GraphKey")
ax.legend()

for group in bars:
    for bar in group:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.show()


# predicted links
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

bars = ax.bar(df["Method"], df["Pred"])

ax.set_yscale("log")
ax.set_ylabel("Predicted links (log scale)")
ax.set_title("Number of predicted links")

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()