import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 模拟注意力分布（文本 token 的注意力）
tokens = ['I', 'feel', 'really', 'tired', 'and', 'empty', 'today', '.', '<PAD>', '<PAD>']
attention_weights = np.array([0.05, 0.1, 0.12, 0.2, 0.08, 0.25, 0.15, 0.05, 0.0, 0.0])

plt.figure(figsize=(10, 2))
sns.heatmap([attention_weights], cmap='YlGnBu', xticklabels=tokens, yticklabels=False, cbar=True, annot=True)
plt.title('Fig.7 - Attention Heatmap (Text Tokens)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualization/figures/fig7_attention_heatmap.png')
print("✅ Fig.7 saved: fig7_attention_heatmap.png")
