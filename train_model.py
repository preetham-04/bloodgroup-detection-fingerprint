import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from resnet9_model import ResNet9
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms including augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder("data/", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load custom ResNet9 model
model = ResNet9(3, 8).to(device)  # 3 input channels, 8 output classes

# Fine-tune all parameters in custom model
for param in model.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
for epoch in range(25):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/25 - Loss: {running_loss:.4f} - Validation Accuracy: {val_acc:.2f}%")

    # Save checkpoint
    checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

# Save final model
torch.save(model.state_dict(), 'resnet9_bloodgroup.pth')
print("🎉 Training complete. Final model saved as 'resnet9_bloodgroup.pth'")