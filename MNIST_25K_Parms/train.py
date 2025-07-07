import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

'''
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''

# has around 9k parameter with 96% accuracy
'''
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 1)  # 1x1 kernel, 8 filters
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 3x3 kernel, 16 filters
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # After two poolings, input size is 28 -> 14 -> 7
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''

'''
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 1)  # 1x1 kernel, 12 filters
        self.conv2 = nn.Conv2d(12, 16, 3, padding=1)  # 3x3 kernel, 16 filters
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)  # 3x3 kernel, 32 filters
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # After three poolings, input size is 28 -> 14 -> 7 -> 3
        self.fc = nn.Linear(32 * 3 * 3, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''

# with  17402 paarmeter & 96 % accuracy
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 1)   # 1x1 kernel, 8 filters
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)  # 3x3 kernel, 8 filters
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)  # 3x3 kernel, 16 filters
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)  # 3x3 kernel, 16 filters
        self.conv5 = nn.Conv2d(16, 24, 3, padding=1)  # 3x3 kernel, 24 filters
        self.conv6 = nn.Conv2d(24, 32, 3, padding=1)  # 3x3 kernel, 32 filters
        self.pool = nn.MaxPool2d(2, 2) # reduces the output chanell by half
        self.relu = nn.ReLU()
        # Pool after conv2, conv4, conv6 (3 times): 28 -> 14 -> 7 -> 3
        self.fc = nn.Linear(32 * 3 * 3, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train():
    device = torch.device('cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    sample_img, _ = train_dataset[0]
    print(f"Input image size: {tuple(sample_img.shape)}")
    model = SimpleCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {num_params}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    losses = []
    accuracies = []
    for epoch in range(1):
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100. * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")
    # Plot loss and accuracy
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, label='Training Loss')
    plt.plot(range(1, len(accuracies)+1), accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.savefig('training_plot.png')
    print('Training plot saved as training_plot.png')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f"model_CPU_{timestamp}.pt")
    print(f"Model saved as model_CPU_{timestamp}.pt")

if __name__ == "__main__":
    train()