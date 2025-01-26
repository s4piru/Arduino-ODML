import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from constants import BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE, DATA_DIR, MODEL_PATH, MEAN, STD


def custom_label_transform(label):
    custom_class_to_idx = {
        "bottle": 0,
        "can": 1
    }
    return custom_class_to_idx[dataset.classes[label]]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # First layer: input channels=3 (RGB), output channels=8
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling
        # Second layer: input channels=8, output channels=16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 32)  # Adjusted for 2 pooling layers
        self.fc2 = nn.Linear(32, num_classes)   # Output layer

    def forward(self, x):
        # After conv1 + pool -> shape: [16, IMG_SIZE/2, IMG_SIZE/2]
        x = self.pool(nn.ReLU()(self.conv1(x)))
        # After conv2 + pool -> shape: [32, IMG_SIZE/4, IMG_SIZE/4]
        x = self.pool(nn.ReLU()(self.conv2(x)))
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer and ReLU activation
        x = nn.ReLU()(self.fc1(x))
        # Apply second fully connected layer (output layer)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    train_dir = os.path.join(DATA_DIR, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    test_dir = os.path.join(DATA_DIR, 'test')
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=2)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    last_10_percent_epoch = int(EPOCHS * 0.9)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Check if current epoch is within the last 10% of training
        if epoch >= last_10_percent_epoch:
            # Save the best model within the last 10% epochs
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Best model updated at epoch {epoch+1} with Val Acc: {val_acc:.2f}%")

    print("Training completed.")

    # Plot Training and Validation Loss
    plt.figure(figsize=(10,5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.png')
    plt.show()
