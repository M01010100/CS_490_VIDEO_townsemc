import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.models as models
from torch.optim import Adam
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(128 * 3 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ResNetCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # First process temporal information
        self.conv3d = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Load pretrained ResNet
        self.resnet = models.resnet18(pretrained=True)
        # Replace first conv layer to handle 64 input channels
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final FC layer
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: [batch, frames, channels, height, width]
        batch_size, frames, channels, height, width = x.size()
        
        # Permute to [batch, channels, frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Apply 3D convolution
        x = self.conv3d(x)
        x = self.bn3d(x)
        x = self.relu(x)
        x = self.pool3d(x)
        
        # Reshape for ResNet
        # Combine batch and temporal dimensions
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, 64, height//2, width//2)
        
        # Pass through ResNet
        x = self.resnet(x)
        
        # Average across temporal dimension
        x = x.view(batch_size, -1, self.resnet.fc.out_features).mean(1)
        
        return x

def get_approach_names():
    return ["SimpleCNN", "ResNetCNN"]

def get_approach_description(approach_name):
    descriptions = {
        "SimpleCNN": "Basic 3D CNN with three convolutional layers",
        "ResNetCNN": "ResNet18 backbone with additional 3D convolution layers"
    }
    return descriptions.get(approach_name, "Unknown approach")

def get_data_transform(approach_name, training):
    base_transform = [
        v2.ToDtype(torch.float32, scale=True)
    ]
    
    if approach_name == "SimpleCNN":
        base_transform.insert(0, v2.Resize((64, 64)))
        if training:
            base_transform.extend([
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15)
            ])
    else:  # ResNetCNN
        base_transform.insert(0, v2.Resize((224, 224)))
        if training:
            base_transform.extend([
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.2, contrast=0.2)
            ])
    
    return v2.Compose(base_transform)

def get_batch_size(approach_name):
    batch_sizes = {
        "SimpleCNN": 32,
        "ResNetCNN": 16  # Smaller batch size due to larger model
    }
    return batch_sizes.get(approach_name, 32)

def create_model(approach_name, class_cnt):
    models = {
        "SimpleCNN": SimpleCNN(class_cnt),
        "ResNetCNN": ResNetCNN(class_cnt)
    }
    return models.get(approach_name)

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 25
    
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            # Handle different dataloader formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    data, target = batch
                elif len(batch) == 3:
                    data, target, _ = batch  # If there's an extra value (like index)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            else:
                raise ValueError("Batch should be a tuple or list")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    # Handle different dataloader formats for test data
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 2:
                            data, target = batch
                        elif len(batch) == 3:
                            data, target, _ = batch
                        else:
                            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                    else:
                        raise ValueError("Batch should be a tuple or list")

                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            print(f'Epoch {epoch}: Test Accuracy: {100 * correct / total:.2f}%')
    
    return model