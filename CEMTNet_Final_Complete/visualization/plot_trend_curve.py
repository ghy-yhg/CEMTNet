import matplotlib.pyplot as plt
import numpy as np

# 模拟情绪趋势数据（Depressed vs Non-depressed）
timesteps = np.arange(10)
depressed = np.array([0.1, 0.15, 0.25, 0.4, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
normal = np.array([0.1, 0.12, 0.14, 0.13, 0.12, 0.13, 0.14, 0.13, 0.14, 0.13])

plt.figure()
plt.plot(timesteps, depressed, label='Depressed', marker='o')
plt.plot(timesteps, normal, label='Non-Depressed', marker='s')
plt.title('Fig.6 - Emotion Trend Over Time')
plt.xlabel('Time Step')
plt.ylabel('Emotion Intensity')
plt.legend()
plt.grid(True)
plt.savefig('visualization/figures/fig6_trend_curve.png')
print("✅ Fig.6 saved: fig6_trend_curve.png")
