import tonic
import matplotlib.pyplot as plt
import numpy as np
import torch

# 0. GPU 设备的自动检测与配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=========================================")
print(f"当前 PyTorch 正在使用的计算设备: {str(device).upper()}")
if device.type == 'cuda':
    print(f"检测到的显卡型号: {torch.cuda.get_device_name(0)}")
print(f"=========================================\n")

print("正在初始化 N-MNIST 数据集...")
# 1. 下载并加载 N-MNIST 数据集
dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
events, target = dataset[0]

print(f"--- 数据集读取成功 ---")
print(f"标签: {target} | 事件总数: {len(events)}")

# 2. 将数据转为 PyTorch 张量并尝试搬运到 GPU 上
# 在实际训练中，这一步是利用显卡加速的核心
event_tensor = torch.tensor(events['t']) # 取出时间戳作为示例
if device.type == 'cuda':
    event_tensor_gpu = event_tensor.to(device)
    print("成功：已将事件时间戳数据载入显存 (GPU Memory)！")

# 3. 简单的 2D 投影累积可视化
sensor_size = dataset.sensor_size
frame = np.zeros((sensor_size[1], sensor_size[0]))

for e in events[:10000]:
    x, y, t, p = e
    frame[y, x] = 1 if p == 1 else -1

# 4. 绘图
plt.figure(figsize=(5, 5))
plt.imshow(frame, cmap='coolwarm')
plt.title(f"N-MNIST Digit: {target} (GPU Ready)")
plt.colorbar(label='Polarity')
print("\n正在弹出可视化窗口...")
plt.show()