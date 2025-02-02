import torch
from torch import nn
from torchvision.transforms import v2
from prettytable import PrettyTable

# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class RNNVideoNet(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()
        self.feature_extract = nn.ModuleList([
            nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(3,3,3), padding="same"), 
            nn.ELU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(8, 16, (3,3,3), padding="same"),
            nn.ELU(),
            nn.MaxPool3d((1,2,2))
        ])
        
        self.flatten = nn.Flatten(start_dim=2)
        
        # New size calculation for 112x112 input
        # After 2 MaxPool3d: 112/4 = 28
        # Last conv has 16 channels
        # So flattened size is: 16 * 28 * 28 = 12544
        expected_size = 12544
        
        self.rnn = nn.RNN(input_size=expected_size, 
                          hidden_size=1024,  
                          num_layers=1,
                          batch_first=True)
        
        self.classifier_stack = nn.Sequential(                           
            nn.Linear(1024, class_cnt)  
        )
        
    def forward(self, x):
        PRINT_DEBUG = True
        print("Input shape:", x.shape)
        x = torch.transpose(x, 1, 2)
        print("After transpose 1:", x.shape)
        
        for index, layer in enumerate(self.feature_extract):
            x = layer(x)
            if PRINT_DEBUG: print(f"After layer {index}:", x.shape)
            
        x = torch.transpose(x, 1, 2)
        print("After transpose 2:", x.shape)
        x = self.flatten(x)
        print("After flatten:", x.shape)
        
        try:
            out, _ = self.rnn(x)
            print("After RNN:", out.shape)
            out = out[:,-1,:]
            print("After selecting last:", out.shape)
            logits = self.classifier_stack(out)
            print("Final output:", logits.shape)
            return logits
        except RuntimeError as e:
            print("Error in RNN forward pass:")
            print(e)
            raise e

def get_approach_names():
    return ["RNN"]

def get_approach_description(approach_name):
    desc = {
        "RNN":"Based on the RNN Video Net example from class, kernel size (3,3,3)"
    }
    return desc.get(approach_name, "Invalid Approach Specified")

def get_data_transform(approach_name, training):
    target_size = (112,112)  #memory constraints for you memeory poor people unlike my 384Gb PowerEdge
    if not training:
        data_transform = v2.Compose([v2.ToImage(), 
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Resize(target_size)])
    else:
        data_transform = v2.Compose([v2.ToImage(), 
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.RandomGrayscale(0.3),
                                    v2.RandomSolarize(0.3),
                                    v2.RandomHorizontalFlip(0.3),
                                    v2.RandomVerticalFlip(0.3),
                                    v2.Resize(target_size)])
    return data_transform

def get_batch_size(approach_name):
    batch_sizes = {
        "RNN": 8  # 8 videos per batch
    }
    return batch_sizes.get(approach_name, None)

def create_model(approach_name, class_cnt):
    model = None
    match approach_name:
        case "RNN":
            model = RNNVideoNet(class_cnt)
        case _:
            print("Invalid Approach Specified")
    return model

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (input, _, label) in enumerate(dataloader):
        input, label = input.to(device), label.to(device)
        pred = model(input)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
            
def test_one_epoch(dataloader, model, loss_fn, data_name, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for input, _, label in dataloader:            
            input, label = input.to(device), label.to(device)
            pred = model(input)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(data_name + f" Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    early_stopper = EarlyStopper(patience=3, min_delta=10) 
    match approach_name:
        case "RNN":
            epochs = 16
        case _:
            epochs = 32
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        validation_loss = test_one_epoch(test_dataloader,model,loss_fn, "Test", device)
        if early_stopper.early_stop(validation_loss):
            print("Stopping early.")           
            break
    return model