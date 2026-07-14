import torch
import torch.nn as nn
import tonic
import tonic.transforms as transforms
import snntorch as snn
from snntorch import functional as SF
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {str(device).upper()}")

# Data pipeline
sensor_size = (34, 34, 2)
time_window = 10000

my_filter = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.Decimation(n=2),
    transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)
])

train_dataset = tonic.datasets.NMNIST(save_to='./data', train=True, transform=my_filter)
test_dataset = tonic.datasets.NMNIST(save_to='./data', train=False, transform=my_filter)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False))

# SNN Model
num_inputs = 34 * 34 * 2
num_hidden = 128
num_outputs = 10
beta = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        
        for step in range(x.size(0)):
            x_flat = x[step].flatten(start_dim=1)
            cur1 = self.fc1(x_flat)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            
        return torch.stack(spk2_rec)

net = Net().to(device)

# Loss & Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
loss_fn = SF.ce_count_loss()

# Train (1 Epoch, 10 batches for quick check)
print("Starting training...")
net.train()
for batch_idx, (data, targets) in enumerate(train_loader):
    data, targets = data.to(device), targets.to(device)
    
    spk_rec = net(data)
    loss = loss_fn(spk_rec, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        break

# Test (1 batch for quick check)
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        spk_rec = net(data)
        correct += SF.accuracy_rate(spk_rec, targets) * targets.size(0)
        total += targets.size(0)
        break

accuracy = (correct / total) * 100
print(f"\nInitial Test Accuracy: {accuracy:.2f}%")