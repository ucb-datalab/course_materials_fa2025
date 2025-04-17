import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models import resnet18

LOCAL = True
CPU = False

torch.set_num_threads(2)  # Control PyTorch thread pool

if CPU:
    device = torch.device("cpu") # For a comparison (for local runs)
else:
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
trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

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
    train_losses, validation_losses, test_accs = [], [], []
    training_start_time = time.time()
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        total = 0
        epoch_start_time = time.time()
        
        if LOCAL:
            from tqdm import tqdm
            data_iterator = tqdm(trainloader)
        else: 
            data_iterator = trainloader
        for data in data_iterator:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)            
            running_loss += loss.item()

        # Save the loss after each epoch
        train_losses.append(running_loss/total)
        
        # Evaluate on test set after each epoch
        validation_loss, accuracy = evaluate_model()
        validation_losses.append(validation_loss)
        test_accs.append(accuracy)
        
        # Save model if it's the best so far, you can also save at the end of training
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_cifar10_model.pth')
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds')
    
    print(f'Finished Training! Best accuracy: {best_acc:.2f}%')
    training_time = time.time() - training_start_time
    print(f'Training completed in {training_time:.2f} seconds')
    return train_losses, validation_losses, test_accs

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    loss = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()

    accuracy = 100 * correct / total
    loss /= total
    print(f'Accuracy on test images: {accuracy:.2f}%')
    return loss, accuracy

# Plot training progress
def plot_results(training_losses, validation_losses, accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training')
    plt.plot(validation_losses, label='Validation')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.savefig('training_results.png')
    plt.close()

training_losses, validation_losses, accuracies = train_model(epochs=5) # Train the model
plot_results(training_losses, validation_losses, accuracies) # Plot results
print("Example completed successfully!")