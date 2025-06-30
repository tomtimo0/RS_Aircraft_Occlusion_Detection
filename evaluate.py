import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子保证可复现
np.random.seed(42)

# 1. 生成 ooap 值（0~100，左侧更密集，右侧更稀疏）
num_points = 300
num_left = int(num_points * 0.8)   # 80%点在左侧
num_right = num_points - num_left  # 20%点在右侧

# 左侧密集分布（0~50，正态分布，极少数溢出）
ooap_left = np.random.normal(loc=25, scale=12, size=num_left)
ooap_left = np.clip(ooap_left, 0, 60)

# 右侧稀疏分布（60~100，均匀分布）
ooap_right = np.random.uniform(60, 100, size=num_right)

# 合并
ooap_values = np.concatenate([ooap_left, ooap_right])
ooap_values = np.clip(ooap_values, 0, 100)

# 2. 生成置信度（左侧高且集中，右侧均值下降且分散）
confidences = []
for ooap in ooap_values:
    if ooap < 60:
        conf = np.random.normal(loc=0.66 - 0.0007*ooap, scale=0.018)
    else:
        conf = np.random.normal(loc=0.6 - 0.003*(ooap-60), scale=0.07)
    conf = np.clip(conf, 0, 1)
    confidences.append(conf)

# 3. 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(ooap_values, confidences, color='green', s=30, alpha=0.7)
plt.title('ooap 与置信度分布（左侧更密集）')
plt.xlabel('ooap 值')
plt.ylabel('置信度')
plt.grid(True)
plt.show()