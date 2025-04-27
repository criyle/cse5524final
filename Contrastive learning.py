import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class Datasets(Dataset):
    def __init__(self, dataset, label_key1='shape', label_key2='color_name'):
        self.dataset = dataset
        self.labels = [f"{sample['config'][label_key1]}-{sample['config'][label_key2]}" for sample in dataset]
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.label_indices = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.label_indices[idx]
        return image, label

class SimCLRDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return self.transform(img), self.transform(img)

class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        base = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, out_dim)

    def forward(self, x):
        x = self.encoder(x).squeeze()
        return F.normalize(self.fc(x), dim=-1)

class ViTEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.fc = nn.Linear(self.vit.num_features, out_dim)

    def forward(self, x):
        x = self.vit(x)
        return F.normalize(self.fc(x), dim=-1)

class SwinEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(self.fc(x), dim=-1)

# ==== Loss ====
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -9e15)
        positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        loss = -positives + torch.logsumexp(sim, dim=1)
        return loss.mean()
def train_simclr(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = model(x1), model(x2)
        loss = loss_fn(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"[SimCLR] Avg Loss: {avg_loss:.4f}")
    return avg_loss

def main(encoder_type='swin'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 50
    out_dim = 64
    temperature = 0.07
    lr = 3e-4

    dataset_hf = load_from_disk('dataset-5000')['train']#this for dataset with 5000 images
    #dataset_hf = load_from_disk('dataset-1000')['train'] # this for dataset with 1000 images.fif you want bigger
    # data only change this part 'dataset-1000'
    base_dataset = Datasets(dataset_hf)

    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor()
    ])
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = SimCLRDataset(base_dataset, transform=simclr_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    loss_fn = NTXentLoss(temperature)

    if encoder_type == 'cnn':
        model = CNNEncoder(out_dim)
    elif encoder_type == 'vit':
        model = ViTEncoder(out_dim)
    elif encoder_type == 'swin':
        model = SwinEncoder(out_dim)
    else:
        raise ValueError("Unsupported encoder_type. Choose from 'cnn', 'vit', 'swin'")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        loss = train_simclr(model, loader, optimizer, loss_fn, device)
        all_losses.append(loss)

    torch.save(model.state_dict(), f"0.simclr_{encoder_type}_encoder.pth")
    print(" Training complete.")
    plt.figure()
    plt.plot(range(1, epochs + 1), all_losses, marker='o')
    plt.title(f"Loss Curve - SIMCLR + {encoder_type.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"loss_curve_simclr_{encoder_type}.png")
    plt.show()
    print("Extracting embeddings for visualization...")
    model.eval()
    transformed_dataset = TransformedDataset(base_dataset, basic_transform)
    vis_loader = DataLoader(transformed_dataset, batch_size=256, shuffle=False, num_workers=4)

    all_embeddings, all_labels = [], []
    for imgs, labels in vis_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            z = model(imgs).cpu()
        all_embeddings.append(z)
        all_labels.extend(labels.numpy())

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = np.array(all_labels)
    tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
    reduced = tsne.fit_transform(all_embeddings)
    angles = np.arctan2(reduced[:, 1], reduced[:, 0])
    sort_idx = np.argsort(angles)
    total = len(reduced)
    circular_x = np.cos(2 * np.pi * np.arange(total) / total)
    circular_y = np.sin(2 * np.pi * np.arange(total) / total)
    circular_points = np.stack([circular_x, circular_y], axis=1)

    shape_map = {
        'circle': 'o',
        'square': 's',
        'triangle': '^'
    }

    raw_colors = [sample['config']['color_name'] for sample in dataset_hf]
    raw_shapes = [sample['config']['shape'] for sample in dataset_hf]

    plt.figure(figsize=(7, 7))
    for i, idx in enumerate(sort_idx):
        x, y = circular_points[i]
        color = raw_colors[idx]
        shape = raw_shapes[idx]
        marker = shape_map.get(shape, 'x')
        plt.scatter(x, y, color=color, marker=marker,
                    edgecolors='black', linewidths=0.2, s=40)

    plt.title("Color-Sensitive Contrastive Embedding")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"tsne_vis_simclr_{encoder_type}_circle_rawcolor.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main(encoder_type='cnn') #this part you could change into "vit" to train ViT and change "swin" to train swin Transformer
