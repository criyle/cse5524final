from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns


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


def loss_function(recon_x, x, mean, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD


def train_vae_model_steps(model: nn.Module, trainloader: DataLoader, testloader: DataLoader, learning_rate=1e-4, weight_decay=1e-5, num_epochs=20):
    # Initialize the model, loss function, and optimizer
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
            outputs, mean, logvar = model(inputs)
            loss = loss_function(outputs, inputs, mean, logvar)

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
                outputs, mean, logvar = model(inputs)
                loss = loss_function(outputs, inputs, mean, logvar)
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
        plt.figure(figsize=(12, 9))
        for i in range(6):
            inputs = testset[i].unsqueeze(0)  # Add batch dimension
            inputs = inputs
            outputs, _ = model(inputs.to(device))
            outputs = outputs.to('cpu')

            # Original Image
            plt.subplot(3, 4, 2*i + 1)
            plt.imshow(inputs.squeeze().permute(1, 2, 0).numpy())
            plt.title('Original Image')

            # Reconstructed Image
            plt.subplot(3, 4, 2*i + 2)
            plt.imshow(outputs.squeeze().permute(1, 2, 0).numpy())
            plt.title('Reconstructed Image')

        plt.tight_layout()
        plt.show()


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


COLORS = ['red', 'orange', 'green', 'purple',
          'blue', 'cyan', 'brown', 'yellow']
SHAPES = ['circle', 'square', 'triangle']
color_mapping = {label: idx for idx, label in enumerate(COLORS)}
shape_mapping = {label: idx for idx, label in enumerate(SHAPES)}


def calc_label(color, shape):
    return color_mapping[color] * len(SHAPES) + shape_mapping[shape]


def calculate_cluster_purity(labels_true, labels_pred):
    contingency_matrix = np.zeros(
        (len(set(labels_true)), len(set(labels_pred))), dtype=int)

    for true_label, pred_label in zip(labels_true, labels_pred):
        contingency_matrix[true_label, pred_label] += 1

    # Calculate purity
    purity = np.sum(np.max(contingency_matrix, axis=0)) / \
        np.sum(contingency_matrix)
    return purity


def fit_kmean(labels_np, embeddings_np):
    kmeans = KMeans(n_clusters=len(set(labels_np)))
    labels_pred = kmeans.fit_predict(embeddings_np)

    # Create a contingency matrix
    contingency_matrix = np.zeros(
        (len(set(labels_np)), len(set(labels_pred))), dtype=int)

    for true_label, pred_label in zip(labels_np, labels_pred):
        contingency_matrix[true_label, pred_label] += 1

    # Use the Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    mapped_labels_pred = np.zeros_like(labels_pred)
    for i, j in zip(row_ind, col_ind):
        mapped_labels_pred[labels_pred == j] = i

    return mapped_labels_pred


def calculate_embeddings(model: nn.Module, testset: Dataset):
    model.eval()
    embeddings = None
    testloader = DataLoader(testset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for data in testloader:
            inputs = data.to(device)
            _, encoded = model(inputs)
            encoded = encoded.to('cpu')
            embeddings = torch.vstack(
                (embeddings, encoded)) if embeddings is not None else encoded
    return embeddings


def calculate_one_metrics(embeddings_np, labels_np):
    labels_pred = fit_kmean(labels_np, embeddings_np)

    sil_score = silhouette_score(embeddings_np, labels_np)
    purity = calculate_cluster_purity(labels_np, labels_pred)
    nmi_score = normalized_mutual_info_score(labels_np, labels_pred)
    conf_matrix = confusion_matrix(labels_np, labels_pred)

    return {
        'sil_score': sil_score,
        'cluster_purity': purity,
        'nmi_score': nmi_score,
        'conf_matrix': conf_matrix,
    }


def calculate_metrics(model: nn.Module, testset: Dataset):
    embeddings = calculate_embeddings(model, testset)
    labels = list([calc_label(x['config']['color_name'], x['config']['shape'])
                  for x in testset.dataset])

    embeddings_np = embeddings.numpy()
    all_labels = np.array(labels)

    color_labels = np.array(
        [color_mapping[x['config']['color_name']] for x in testset.dataset])

    shape_labels = np.array([shape_mapping[x['config']['shape']]
                            for x in testset.dataset])

    return {
        'all': calculate_one_metrics(embeddings_np, all_labels),
        'color': calculate_one_metrics(embeddings_np, color_labels),
        'shape': calculate_one_metrics(embeddings_np, shape_labels),
    }


def print_metrics(metrics):
    print(f"Silhouette Score: {metrics['sil_score']}")
    print(f"Cluster Purity: {metrics['cluster_purity']}")
    print(f"Normalized Mutual Information (NMI): {metrics['nmi_score']}")


def calc_and_plot_metrics(model: nn.Module, testset: Dataset):
    metrics = calculate_metrics(model, testset)
    print_metrics(metrics['all'])
    plot_conf_matrix(metrics['all']['conf_matrix'])
    print_metrics(metrics['color'])
    plot_conf_matrix(metrics['color']['conf_matrix'])
    print_metrics(metrics['shape'])
    plot_conf_matrix(metrics['shape']['conf_matrix'])


def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def save_to_file(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def load_from_file(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def calc_and_plot_conf_matrix(label, pred):
    contingency_matrix = np.zeros(
        (len(set(label)), len(set(pred))), dtype=int)

    for true_label, pred_label in zip(label, pred):
        contingency_matrix[true_label, pred_label] += 1

    plot_conf_matrix(contingency_matrix)


PLOT_SHAPE = ['o', 's', '^']

# t-SNE Visualization


def tsne_visualization(embeddings, colors, shapes):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    colors = np.array([COLORS[x] for x in colors])
    shapes = np.array([PLOT_SHAPE[x] for x in shapes])

    plt.figure(figsize=(10, 10))
    unique_combinations = set(zip(colors, shapes))
    for color, shape in unique_combinations:
        indices = (colors == color) & (shapes == shape)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    c=color, marker=shape, label=f'{color}-{shape}')
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.legend()
    plt.show()

# PCA Visualization


def pca_visualization(embeddings, colors, shapes):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings)
    colors = np.array([COLORS[x] for x in colors])
    shapes = np.array([PLOT_SHAPE[x] for x in shapes])

    plt.figure(figsize=(10, 10))
    unique_combinations = set(zip(colors, shapes))
    for color, shape in unique_combinations:
        indices = (colors == color) & (shapes == shape)
        plt.scatter(pca_results[indices, 0], pca_results[indices, 1],
                    c=color, marker=shape, label=f'{color}-{shape}')
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.legend()
    plt.show()


def fit_nearest_neighbor(model):
    training_set = GeneratedDataset('dataset-1000', transforms.ToTensor())
    test_set = GeneratedDataset('dataset-test', transforms.ToTensor())

    train_embeddings = calculate_embeddings(model, training_set)
    test_embeddings = calculate_embeddings(model, test_set)

    # Convert embeddings to tensors
    train_embeddings = torch.tensor(train_embeddings)
    test_embeddings = torch.tensor(test_embeddings)

    # Initialize NearestNeighbors model
    nbrs = NearestNeighbors(
        n_neighbors=5, algorithm='auto').fit(train_embeddings)

    # Find the nearest neighbors for each test embedding
    distances, indices = nbrs.kneighbors(test_embeddings)

    # Match shape & color & all
    color = 0
    shape = 0
    all = 0
    color_label = []
    color_pred = []
    shape_label = []
    shape_pred = []
    for i in range(len(test_set)):
        t_shape = test_set.dataset[i]['config']['shape']
        t_color = test_set.dataset[i]['config']['color_name']

        p_shape = training_set.dataset[int(indices[i, 0])]['config']['shape']
        p_color = training_set.dataset[int(
            indices[i, 0])]['config']['color_name']

        if t_shape == p_shape:
            shape += 1
        if t_color == p_color:
            color += 1
        if t_shape == p_shape and t_color == p_color:
            all += 1

        color_label.append(color_mapping[t_color])
        color_pred.append(color_mapping[p_color])
        shape_label.append(shape_mapping[t_shape])
        shape_pred.append(shape_mapping[p_shape])

    print(f'Color Accuracy: {color / len(test_set)}')
    print(f'Shape Accuracy: {shape / len(test_set)}')
    print(f'Combined Accuracy: {all / len(test_set)}')

    # Plot the images of the top 5 nearest neighbors for each test embedding
    plt.figure(figsize=(12, 10))
    for i in range(5):
        plt.subplot(5, 6, i*6 + 1)
        plt.title(f'Example {i+1}')
        plt.axis('off')
        plt.imshow(test_set.dataset[int(i)]['image'])
        for j, idx in enumerate(indices[i]):
            plt.subplot(5, 6, i*6 + j+2)
            plt.imshow(training_set.dataset[int(idx)]['image'])
            plt.title(f'Neighbor {j+1}')
            plt.axis('off')
    plt.show()

    tsne_visualization(test_embeddings, color_label, shape_label)
    pca_visualization(test_embeddings, color_label, shape_label)

    calc_and_plot_conf_matrix(color_label, color_pred)
    calc_and_plot_conf_matrix(shape_label, shape_pred)
    return (color, shape, all)
