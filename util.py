import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim


class GeneratedDataset(Dataset):
    def __init__(self, name, transform=None):
        self.dataset_dict = load_from_disk(name)
        self.dataset = self.dataset_dict['train']
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return image


device = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'


def train_model_steps(model: nn.Module, trainloader: DataLoader, testloader: DataLoader, learning_rate=1e-4, weight_decay=1e-5, num_epochs=20):
    # Initialize the model, loss function, and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)

    # Arrays to store training and testing loss
    train_losses = []
    test_losses = []

    model = model.to(device)

    # Training the autoencoder
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in trainloader:
            inputs = data.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in testloader:
                inputs = data.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()

        test_loss /= len(testloader)
        test_losses.append(test_loss)

        print(
            f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        scheduler.step()

    # Save the model
    return model, train_losses, test_losses


def plot_model(model, train_losses, test_losses, testset):
    # Plot the training and testing loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    plt.show()

    # Show some examples of reconstruction
    model.eval()
    with torch.no_grad():
        for i in range(5):  # Show 5 examples
            inputs = testset[i].unsqueeze(0)  # Add batch dimension
            inputs = inputs
            outputs, _ = model(inputs.to(device))
            outputs = outputs.to('cpu')

            plt.figure(figsize=(6, 3))

            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(inputs.squeeze().permute(1, 2, 0).numpy())
            plt.title('Original Image')

            # Reconstructed Image
            plt.subplot(1, 2, 2)
            plt.imshow(outputs.squeeze().permute(1, 2, 0).numpy())
            plt.title('Reconstructed Image')

            plt.show()


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
