import matplotlib.pyplot as plt

# 模拟模块消融实验数据（实际使用可从日志中加载）
modules = ['Full Model', '-CCIM', '-EMSA', '-CE-TM']
f1_scores = [0.93, 0.88, 0.89, 0.87]

plt.figure()
plt.bar(modules, f1_scores, color='skyblue')
plt.title('Fig.5 - Ablation Study (F1 Score)')
plt.ylabel('F1 Score')
plt.ylim(0.80, 0.95)
plt.savefig('visualization/figures/fig5_ablation_bar.png')
print("✅ Fig.5 saved: fig5_ablation_bar.png")
