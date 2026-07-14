import tonic
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {str(device).upper()}")

print("Loading N-MNIST dataset...")

# Load Raw Dataset
dataset_raw = tonic.datasets.NMNIST(save_to='./data', train=True)
events_raw, target = dataset_raw[0]
raw_count = len(events_raw)

# Define Filters
my_filter = transforms.Compose([
    transforms.Denoise(filter_time=10000), 
    transforms.Decimation(n=2)            
])

# Load Filtered Dataset
dataset_filtered = tonic.datasets.NMNIST(save_to='./data', train=True, transform=my_filter)
events_filtered, _ = dataset_filtered[0]
filtered_count = len(events_filtered)

# Print Statistics
reduction_rate = (1 - filtered_count / raw_count) * 100
print(f"--- Dataset Comparison ---")
print(f"Label: {target}")
print(f"Raw Events: {raw_count}")
print(f"Filtered Events: {filtered_count}")
print(f"Data Reduction Rate: {reduction_rate:.2f}%")

# Build 2D Frame for Visualization
sensor_size = dataset_filtered.sensor_size
frame = np.zeros((sensor_size[1], sensor_size[0]))

for e in events_filtered:
    x, y, p = e['x'], e['y'], e['p']
    frame[y, x] = 1 if p == 1 else -1

# Plotting
plt.figure(figsize=(5, 5))
plt.imshow(frame, cmap='coolwarm')
plt.title(f"N-MNIST Digit: {target} (Filtered)")
plt.colorbar(label='Polarity')
plt.show()