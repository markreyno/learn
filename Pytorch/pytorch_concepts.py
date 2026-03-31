# ============================================================
# PYTORCH MAIN CONCEPTS
# ============================================================
# PyTorch is a deep learning framework built around dynamic computation
# graphs. Core ideas: Tensors (GPU-accelerated arrays), autograd
# (automatic differentiation), and nn.Module (model building blocks).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np

# ============================================================
# 1. TENSORS
# ============================================================
# Tensors are the fundamental data structure — like NumPy arrays
# but with GPU support and autograd integration.

print("--- Tensors ---")

# Creating tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(torch.zeros(3, 4))
print(torch.ones(2, 3))
print(torch.full((2, 2), 7.0))
print(torch.arange(0, 10, 2))
print(torch.linspace(0, 1, 5))
print(torch.rand(3, 3))          # uniform [0, 1)
print(torch.randn(3, 3))         # standard normal
print(torch.eye(3))              # identity matrix

# Tensor attributes
t = torch.randn(4, 5)
print("shape  :", t.shape)        # torch.Size([4, 5])
print("dtype  :", t.dtype)        # torch.float32
print("device :", t.device)       # cpu or cuda:0
print("ndim   :", t.ndim)

# Specifying dtype
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x = x.to(torch.float64)           # cast
x = x.int()                       # shorthand cast


# ============================================================
# 2. TENSOR OPERATIONS
# ============================================================

print("\n--- Tensor Operations ---")

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print(a + b)                      # element-wise add
print(a * b)                      # element-wise multiply
print(a @ b)                      # matrix multiply
print(torch.matmul(a, b))         # same as @

print(a.sum())                    # scalar sum
print(a.sum(dim=0))               # column sums
print(a.sum(dim=1))               # row sums
print(a.mean(), a.std())
print(a.max(), a.argmax())
print(a.T)                        # transpose

# In-place operations (trailing _)
a.add_(1)                         # a += 1  (modifies a)
a.mul_(2)                         # a *= 2


# ============================================================
# 3. RESHAPING & INDEXING
# ============================================================

print("\n--- Reshaping & Indexing ---")

t = torch.arange(12, dtype=torch.float32)
print(t.reshape(3, 4))
print(t.view(4, 3))               # view: shares memory (must be contiguous)
print(t.reshape(2, 2, 3))

# Squeeze / unsqueeze
t = torch.randn(3, 1, 4)
print(t.squeeze())                # remove dims of size 1 -> (3, 4)
print(t.unsqueeze(0).shape)       # add dim at position 0 -> (1, 3, 1, 4)

# Indexing (same syntax as NumPy)
t = torch.arange(9).reshape(3, 3)
print(t[0])                       # first row
print(t[:, 1])                    # second column
print(t[1:, :2])                  # slice

# Boolean indexing
print(t[t > 4])

# Stack / cat
a = torch.ones(3)
b = torch.zeros(3)
print(torch.cat([a, b], dim=0))   # concatenate along existing dim
print(torch.stack([a, b], dim=0)) # stack into new dim


# ============================================================
# 4. GPU / DEVICE MANAGEMENT
# ============================================================

print("\n--- Device ---")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

t = torch.randn(3, 3)
t = t.to(device)                  # move to GPU (or stay on CPU)

# Create directly on device
t_gpu = torch.zeros(3, 3, device=device)

# Move back to CPU for NumPy interop
t_cpu = t.cpu()


# ============================================================
# 5. AUTOGRAD — AUTOMATIC DIFFERENTIATION
# ============================================================
# PyTorch tracks operations on tensors with requires_grad=True
# and computes gradients via backpropagation.

print("\n--- Autograd ---")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 3 * x               # y = x³ + 3x
y.backward()                      # compute dy/dx
print("x:", x)
print("dy/dx at x=2:", x.grad)   # 3x² + 3 = 15

# Multi-variable example
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
loss = ((a * b) ** 2).sum()
loss.backward()
print("grad a:", a.grad)
print("grad b:", b.grad)

# Detach from graph (no gradient tracking)
with torch.no_grad():             # preferred: disables grad tracking
    z = x * 2
z2 = x.detach() * 2              # alternative: detach tensor


# ============================================================
# 6. BUILDING MODELS WITH nn.Module
# ============================================================
# Every model is a subclass of nn.Module.
# Define layers in __init__, forward pass in forward().

print("\n--- nn.Module ---")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)           # raw logits (no activation — handled by loss)
        return x

model = MLP(input_size=10, hidden_size=64, output_size=3)
print(model)

# Inspect parameters
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,} | Trainable: {trainable:,}")

# Quick forward pass check
dummy_input = torch.randn(8, 10)   # batch of 8 samples, 10 features
output = model(dummy_input)
print("Output shape:", output.shape)  # (8, 3)

# Sequential — shorthand for simple stacks
simple_model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)


# ============================================================
# 7. COMMON LAYERS
# ============================================================

print("\n--- Common Layers ---")

# Linear (fully connected)
fc = nn.Linear(in_features=128, out_features=64, bias=True)

# Convolutional
conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                   stride=1, padding=1)
pool   = nn.MaxPool2d(kernel_size=2, stride=2)

# Recurrent
rnn  = nn.RNN(input_size=10, hidden_size=32, num_layers=1, batch_first=True)
lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2,
               batch_first=True, dropout=0.2)
gru  = nn.GRU(input_size=10, hidden_size=32, batch_first=True)

# Normalization
bn1d = nn.BatchNorm1d(num_features=64)
bn2d = nn.BatchNorm2d(num_features=32)
ln   = nn.LayerNorm(normalized_shape=64)

# Regularization
dropout  = nn.Dropout(p=0.5)
dropout2 = nn.Dropout2d(p=0.5)     # for spatial (image) features

# Embedding (for NLP / categorical inputs)
emb = nn.Embedding(num_embeddings=1000, embedding_dim=64)
x_idx = torch.tensor([1, 5, 42])
print("Embedding output:", emb(x_idx).shape)   # (3, 64)


# ============================================================
# 8. ACTIVATION FUNCTIONS
# ============================================================

print("\n--- Activations ---")

x = torch.linspace(-3, 3, 7)
print("ReLU    :", F.relu(x))
print("LeakyReLU:", F.leaky_relu(x, negative_slope=0.1))
print("Sigmoid :", torch.sigmoid(x))
print("Tanh    :", torch.tanh(x))
print("Softmax :", F.softmax(x, dim=0))
print("GELU    :", F.gelu(x))
print("SiLU    :", F.silu(x))       # used in modern transformers

# As nn.Module layers (useful inside nn.Sequential)
nn.ReLU(), nn.LeakyReLU(0.1), nn.Sigmoid(), nn.Tanh(), nn.GELU()


# ============================================================
# 9. LOSS FUNCTIONS
# ============================================================

print("\n--- Loss Functions ---")

# Classification
logits   = torch.randn(8, 3)        # raw model output (no softmax)
targets  = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

ce_loss  = nn.CrossEntropyLoss()(logits, targets)  # multi-class
print("CrossEntropyLoss:", ce_loss.item())

bce_loss = nn.BCEWithLogitsLoss()   # binary classification (applies sigmoid internally)

# Regression
preds    = torch.randn(8)
labels   = torch.randn(8)
mse_loss = nn.MSELoss()(preds, labels)
mae_loss = nn.L1Loss()(preds, labels)
huber    = nn.HuberLoss()(preds, labels)   # robust to outliers
print("MSELoss:", mse_loss.item())


# ============================================================
# 10. OPTIMIZERS
# ============================================================

print("\n--- Optimizers ---")

model = MLP(10, 64, 3)

# Common optimizers
sgd    = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                   weight_decay=1e-4)
adam   = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                    weight_decay=1e-4)
adamw  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
rmsprop = optim.RMSprop(model.parameters(), lr=1e-3)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.5)
cosine_lr = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=50)
plateau   = optim.lr_scheduler.ReduceLROnPlateau(adam, patience=5,
                                                   factor=0.5)
warmup    = optim.lr_scheduler.LinearLR(adam, start_factor=0.1,
                                         total_iters=5)


# ============================================================
# 11. DATASETS & DATALOADERS
# ============================================================

print("\n--- Dataset & DataLoader ---")

# TensorDataset — wraps tensors directly
X_data = torch.randn(200, 10)
y_data = torch.randint(0, 3, (200,))
dataset = TensorDataset(X_data, y_data)

# Split into train/val
train_ds, val_ds = random_split(dataset, [160, 40])

# DataLoader — batching, shuffling, parallel loading
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

print("Batches in train:", len(train_loader))
for X_batch, y_batch in train_loader:
    print("Batch shape:", X_batch.shape, y_batch.shape)
    break

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[idx]

custom_ds = MyDataset(X_data, y_data)
loader    = DataLoader(custom_ds, batch_size=16, shuffle=True)


# ============================================================
# 12. TRAINING LOOP
# ============================================================

print("\n--- Training Loop ---")

model     = MLP(10, 64, 3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()                           # enable dropout, batchnorm training mode
    total_loss, correct = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()               # clear previous gradients
        outputs = model(X_batch)            # forward pass
        loss    = criterion(outputs, y_batch)
        loss.backward()                     # compute gradients
        optimizer.step()                    # update weights

        total_loss += loss.item() * len(X_batch)
        correct    += (outputs.argmax(1) == y_batch).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()                            # disable dropout, use running stats
    total_loss, correct = 0.0, 0
    with torch.no_grad():                   # no gradient computation needed
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            total_loss += loss.item() * len(X_batch)
            correct    += (outputs.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Full training loop
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                             criterion, device)
    val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")


# ============================================================
# 13. CONVOLUTIONAL NEURAL NETWORK (CNN)
# ============================================================

print("\n--- CNN ---")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 32, H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 64, H/4, W/4)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cnn = CNN(num_classes=10)
dummy_img = torch.randn(4, 1, 28, 28)   # batch of 4 grayscale 28×28 images
print("CNN output shape:", cnn(dummy_img).shape)  # (4, 10)


# ============================================================
# 14. RECURRENT NEURAL NETWORK (RNN / LSTM)
# ============================================================

print("\n--- LSTM ---")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])   # use last timestep
        return out

lstm_model = LSTMClassifier(input_size=10, hidden_size=64,
                              num_layers=2, num_classes=3)
seq_input = torch.randn(8, 20, 10)    # batch=8, seq_len=20, features=10
print("LSTM output shape:", lstm_model(seq_input).shape)  # (8, 3)


# ============================================================
# 15. SAVING & LOADING MODELS
# ============================================================

print("\n--- Save & Load ---")

# Save/load only weights (recommended)
torch.save(model.state_dict(), "model_weights.pth")

loaded_model = MLP(10, 64, 3)
loaded_model.load_state_dict(torch.load("model_weights.pth",
                                         map_location=device))
loaded_model.eval()
print("Model loaded successfully")

# Save/load full model (less portable — ties to class definition)
torch.save(model, "full_model.pth")
full = torch.load("full_model.pth", map_location=device)

# Save training checkpoint (resume training)
torch.save({
    "epoch":           EPOCHS,
    "model_state":     model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_loss":        val_loss,
}, "checkpoint.pth")

checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_epoch = checkpoint["epoch"]

# Clean up demo files
import os
for f in ["model_weights.pth", "full_model.pth", "checkpoint.pth"]:
    os.remove(f)


# ============================================================
# 16. TRANSFER LEARNING
# ============================================================
# Use a pretrained model as a feature extractor or fine-tune it.

print("\n--- Transfer Learning ---")
import torchvision.models as models   # requires torchvision

# Load a pretrained ResNet-18
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers
for param in backbone.parameters():
    param.requires_grad = False

# Replace the final classification head
num_features = backbone.fc.in_features        # 512
backbone.fc  = nn.Linear(num_features, 5)     # new task: 5 classes

# Only the new head trains
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"Trainable params after freezing: {trainable:,}")

# Fine-tune: unfreeze later layers
for param in backbone.layer4.parameters():
    param.requires_grad = True

optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, backbone.parameters()),
    lr=1e-4,
)


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# TENSORS
#   torch.tensor([...])                  Create from data
#   torch.zeros/ones/rand/randn(shape)   Constant / random
#   torch.arange / torch.linspace        Range / evenly spaced
#   t.shape / t.dtype / t.device         Attributes
#   t.to(device) / t.cpu() / t.cuda()    Move device
#   t.float() / t.int() / t.to(dtype)    Cast
#
# OPERATIONS
#   a + b / a * b / a @ b                Element-wise / matmul
#   t.sum/mean/max/min(dim=)             Reductions
#   t.reshape / t.view / t.T             Reshape / transpose
#   t.squeeze() / t.unsqueeze(dim)       Add/remove dims
#   torch.cat([a,b], dim=) / torch.stack Stack tensors
#
# AUTOGRAD
#   tensor(..., requires_grad=True)      Track gradients
#   loss.backward()                      Compute gradients
#   optimizer.zero_grad()                Clear gradients
#   with torch.no_grad():                Disable tracking
#   tensor.detach()                      Detach from graph
#
# MODELS
#   class Model(nn.Module)               Custom model
#   nn.Sequential(...)                   Simple stack
#   model.train() / model.eval()         Toggle train/eval mode
#   model.parameters()                   All learnable params
#   model.state_dict()                   Weight dictionary
#
# COMMON LAYERS
#   nn.Linear(in, out)                   Fully connected
#   nn.Conv2d(in_ch, out_ch, ks)         2D convolution
#   nn.LSTM(input, hidden, layers)       LSTM
#   nn.Embedding(vocab, dim)             Embedding table
#   nn.BatchNorm1d/2d / nn.LayerNorm     Normalization
#   nn.Dropout(p)                        Regularization
#
# LOSS FUNCTIONS
#   nn.CrossEntropyLoss()                Multi-class classification
#   nn.BCEWithLogitsLoss()               Binary classification
#   nn.MSELoss() / nn.L1Loss()           Regression
#   nn.HuberLoss()                       Robust regression
#
# OPTIMIZERS
#   optim.SGD / Adam / AdamW             Gradient descent variants
#   scheduler.step()                     Update learning rate
#   ReduceLROnPlateau / CosineAnnealingLR Adaptive schedules
#
# DATA
#   TensorDataset(X, y)                  Wrap tensors
#   random_split(dataset, [n1, n2])      Train/val split
#   DataLoader(ds, batch_size, shuffle)  Batching & sampling
#   class MyDataset(Dataset)             Custom dataset
#
# TRAINING LOOP
#   model.train()                        Enable dropout/batchnorm
#   optimizer.zero_grad()                1. Clear grads
#   output = model(X)                    2. Forward pass
#   loss = criterion(output, y)          3. Compute loss
#   loss.backward()                      4. Backprop
#   optimizer.step()                     5. Update weights
#   model.eval() + torch.no_grad()       Evaluation mode
#
# SAVE & LOAD
#   torch.save(model.state_dict(), f)    Save weights
#   model.load_state_dict(torch.load(f)) Load weights
#   torch.save({epoch, state,...}, f)    Save checkpoint
#
# TRANSFER LEARNING
#   models.resnet18(weights=...)         Load pretrained
#   param.requires_grad = False          Freeze layer
#   model.fc = nn.Linear(in, out)        Replace head
