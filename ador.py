import matplotlib.pyplot as plt
import numpy as np

# =========================
# Deployment Phase Data
# =========================
deployment_metrics = [
    "Loading Time\n(s)",
    "File Size\n(MB)",
    "Memory Increase\n(MB)",
    "Avg Temp\n(°C)"
]

pytorch_deploy = [0.2177, 1.1284, 13.03, 58.75]
libtorch_deploy = [0.2747, 1.1342, 19.56, 57.30]

# =========================
# Prediction Phase Data
# =========================
prediction_metrics = [
    "Accuracy\n(%)",
    "Prediction Time\n(s)",
    "Latency\n(ms/img)",
    "Throughput\n(img/s)",
    "CPU Usage\n(%)",
    "Memory Usage\n(%)",
    "Avg Temp\n(°C)",
    "GCF Score"
]

pytorch_pred = [100.00, 0.3547, 3.55, 281.94, 34.25, 14.88, 59.30, 0.1966]
libtorch_pred = [98.00, 0.4626, 4.63, 216.18, 29.25, 12.39, 56.45, 0.1977]

# =========================
# Figure 1: Deployment Phase
# =========================
x = np.arange(len(deployment_metrics))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, pytorch_deploy, width, label='PyTorch')
bars2 = plt.bar(x + width/2, libtorch_deploy, width, label='LibTorch')

plt.title("Deployment Phase Comparison", fontsize=16, fontweight='bold')
plt.ylabel("Value", fontsize=12)
plt.xticks(x, deployment_metrics, fontsize=11)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add values on bars
for bar in list(bars1) + list(bars2):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.savefig("deployment_phase_bar_chart.png", dpi=300)
plt.show()

# =========================
# Figure 2: Prediction Phase
# =========================
x = np.arange(len(prediction_metrics))

plt.figure(figsize=(14, 6))
bars1 = plt.bar(x - width/2, pytorch_pred, width, label='PyTorch')
bars2 = plt.bar(x + width/2, libtorch_pred, width, label='LibTorch')

plt.title("Prediction Phase Comparison", fontsize=16, fontweight='bold')
plt.ylabel("Value", fontsize=12)
plt.xticks(x, prediction_metrics, fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add values on bars
for bar in list(bars1) + list(bars2):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=8,
        rotation=90
    )

plt.tight_layout()
plt.savefig("prediction_phase_bar_chart.png", dpi=300)
plt.show()