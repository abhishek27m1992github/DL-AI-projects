import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
from train import SimpleCNN

def test_parameter_count():
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Parameter count too high: {total_params}"
    print(f"Parameter count test passed: {total_params}")

def test_input_output_shape():
    model = SimpleCNN()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Output shape incorrect: {output.shape}"
    print(f"Input/output shape test passed: {output.shape}")

def test_accuracy():
    device = torch.device('cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    model = SimpleCNN().to(device)
    # Find latest model
    model_files = sorted(glob.glob('model_CPU_*.pt'))
    assert model_files, "No model file found. Run train.py first."
    model.load_state_dict(torch.load(model_files[-1], map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    acc = correct / total
    print(f"Test accuracy: {acc*100:.2f}%")
    assert acc > 0.95, f"Accuracy too low: {acc*100:.2f}%"
    print("Accuracy test passed.")

def main():
    test_parameter_count()
    test_input_output_shape()
    test_accuracy()

if __name__ == "__main__":
    main() 