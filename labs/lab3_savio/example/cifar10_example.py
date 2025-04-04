import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import resnet18

torch.set_num_threads(2)  # Control PyTorch thread pool

# device = torch.device("cpu") # For a comparison (for local runs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Load CIFAR-10 dataset
print("Loading datasets...")
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize model
print("Initializing model...")
model = resnet18(weights=None)  # No pre-training
model.fc = nn.Linear(model.fc.in_features, 10) # Adjust the final layer for CIFAR-10's 10 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Training function
def train_model(epochs):
    best_acc = 0.0
    train_losses, test_accs = [], []
    training_start_time = time.time()
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        for data in tqdm(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Save the loss after each epoch
        train_losses.append(running_loss)
        
        # Evaluate on test set after each epoch
        accuracy = evaluate_model()
        test_accs.append(accuracy)
        
        # Save model if it's the best so far
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_cifar10_model.pth')
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds')
    
    print(f'Finished Training! Best accuracy: {best_acc:.2f}%')
    training_time = time.time() - training_start_time
    print(f'Training completed in {training_time:.2f} seconds')
    return train_losses, test_accs

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy:.2f}%')
    return accuracy

# Class-wise accuracy
def class_accuracy():
    model.eval()
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(4, labels.size(0)):  # Get stats for 4 classes at a time
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

# Plot training progress
def plot_results(losses, accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.savefig('training_results.png')
    plt.close()

# Train the model
losses, accuracies = train_model(epochs=3)

# Plot results
plot_results(losses, accuracies)

# Get class-wise accuracy
class_accuracy()

print("Example completed successfully!")