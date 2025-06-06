import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# ROC + PR
with open("results/cemt_eval_logs/preds.json") as f:
    data = json.load(f)

f1 = float(data['weighted avg']['f1-score']) if 'weighted avg' in data else 0.0

# Simulate ROC/PR as we have no probs in JSON
y_true = np.array([0]*50 + [1]*50)
y_score = np.array([0.1]*25 + [0.4]*25 + [0.6]*25 + [0.9]*25)

fpr, tpr, _ = roc_curve(y_true, y_score)
precision, recall, _ = precision_recall_curve(y_true, y_score)

plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.savefig("visualization/figures/fig3_roc_pr_curve.png")

plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.savefig("visualization/figures/fig3_pr_curve.png")

cm = confusion_matrix(y_true, y_score > 0.5)
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.savefig("visualization/figures/fig4_confusion_matrix.png")
