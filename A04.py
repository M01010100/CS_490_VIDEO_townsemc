import torch
import torchvision.transforms as v2

def get_approach_names():
    return ["CNN0", "CNN1"]

def get_approach_description(approach_name):
    descriptions = {
        "CNN0": "A simple CNN with basic layers.",
        "CNN1": "A more complex CNN with additional layers."
    }
    return descriptions.get(approach_name, "Unknown approach")

def get_data_transform(approach_name, training):
    def normalize_video(x):
        # x is expected to be [C, F, H, W]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        x = x.permute(1, 0, 2, 3)  # [F, C, H, W] -> [C, F, H, W]
        x = (x - mean) / std
        return x

    if training:
        return v2.Compose([
            v2.Resize((240, 352)),
            v2.Lambda(lambda x: x.float() / 255.0),  # Normalize to [0,1]
            v2.Lambda(normalize_video)  # Will output [C, F, H, W]
        ])
    else:
        return v2.Compose([
            v2.Resize((240, 352)),
            v2.Lambda(lambda x: x.float() / 255.0),
            v2.Lambda(normalize_video)
        ])

def get_batch_size(approach_name):
    batch_sizes = {
        "CNN0": 32,
        "CNN1": 16
    }
    return batch_sizes.get(approach_name, 32)

def create_model(approach_name, class_cnt):
    # Calculate proper dimensions based on input size
    frames, height, width = 30, 240, 352
    if approach_name == "CNN0":
        # Adjust dimensions after pooling operations
        f_out = frames // 4  # After 2 pooling layers
        h_out = height // 4
        w_out = width // 4
        model = torch.nn.Sequential(
            torch.nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * f_out * h_out * w_out, class_cnt)
        )
    elif approach_name == "CNN1":
        # Adjust dimensions after pooling operations
        f_out = frames // 8  # After 3 pooling layers
        h_out = height // 8
        w_out = width // 8
        model = torch.nn.Sequential(
            torch.nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * f_out * h_out * w_out, class_cnt)
        )
    return model

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].squeeze().long().to(device)  # Squeeze labels to remove extra dimensions
            
            # Clear memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch[0].to(device)
                labels = batch[1].squeeze().long().to(device)  # Squeeze labels here too
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Clear memory
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

        print(f'Epoch {epoch+1}/{num_epochs}, Accuracy: {100 * correct / total}%')

    return model