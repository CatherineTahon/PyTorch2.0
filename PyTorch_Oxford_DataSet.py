import random
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict

# ------------------- Calculation -------------------
stats_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

stats_dataset = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=False,
    transform=stats_transform
)

stats_loader = DataLoader(
    stats_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

channel_sum = torch.zeros(3)
channel_squared_sum = torch.zeros(3)
num_pixels = 0

for images, _ in stats_loader:
    channel_sum += images.sum(dim=[0, 2, 3])
    channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

    num_pixels += images.size(0) * images.size(2) * images.size(3)

mean = channel_sum / num_pixels

std = torch.sqrt(
    channel_squared_sum / num_pixels - mean ** 2
)

print("Mean:", mean)
print("Std:", std)

# ------------------- Helpers -------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ------------------- Device -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ------------------- Data -------------------
transform_train = transforms.Compose([
    #without the data augmentation the data was overfitting! I used a small change because otherwise the model would underfit
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.4783, 0.4459, 0.3957), (0.2599, 0.2545, 0.2625)),
])

transform_val = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.4783, 0.4459, 0.3957), (0.2599, 0.2545, 0.2625)),
])

train_ds = datasets.OxfordIIITPet(root="./data", split="trainval", target_types="category", download=True, transform=transform_train)
val_ds = datasets.OxfordIIITPet(root="./data", split="test", target_types="category", download=True, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=True)

print(len(train_ds))
image,label = train_ds[0]
print(image.shape)
print(label)
print(train_ds.classes[label])
num_classes = len(train_ds.classes)
print(num_classes)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

class PetCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            # input rgb = 3, output = 16, filtersize 3x3
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            # size 128*128 128-3+1=126
            nn.MaxPool2d(2),
            # 126/2=63
            # input 16, output = 64, filtersize 3x3
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            # 64*32*32
            nn.MaxPool2d(2),
            # 64*16*16
            # input 64, output = 128, filtersize 3x3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            # 128*16*16
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = PetCNN(num_classes).to(device)

x = torch.randn(32, 3, 128, 128).to(device)

output = model(x)

print(output.shape)

# ------------------- Loss, Optimizer, Scheduler -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, grad_clip=None):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='val', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += float(loss.item()) * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

# ------------------- Training loop -------------------
num_epochs = 20
scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
best_val_acc = 0.0
history = defaultdict(list)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, grad_clip=5.0)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f" train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}")
    print(f" val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}, 'oxford_best_checkpoint.pth')
        print(' Saved best checkpoint')

print('Training finished')

#sidenote: this model is overfitting