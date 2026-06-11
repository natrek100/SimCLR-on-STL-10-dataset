# =====================================================================
# 1. IMPORTS & DEPENDENCIES
# =====================================================================
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_directml
import torchvision.models as models
import umap
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# =====================================================================
# 2. ENVIRONMENT SETUP
# =====================================================================
@dataclass
class Config:
    seed: int = 42
    batch_size: int = 128
    device: torch.device = torch_directml.device()
    num_epochs: int = 10
    lr: float = 1e-3
    knn_k: int = 15
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

cfg = Config()


def set_seed(seed: int = 42):
    """Sets random seeds across all platforms for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 3. DATASETS & ARCHITECTURES
# =====================================================================
class SimCLR_aug_dataset:
    """Wraps a standard dataset to output two distinct augmented views of the same image."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        aug1 = self.transform(img)
        aug2 = self.transform(img)
        return aug1, aug2


class SimCLREncoder(nn.Module):
    """ResNet18-based Encoder with a 3-layer projection MLP head."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18()

        #extract features without the final classification layer - backbone that learned but did not perform the final classification
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        #projection Head (z)
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.flatten(start_dim=1)
        z = self.projector(h)
        return h, z


class LinearClassifier(nn.Module):
    """Linear probe wrapper attaching a single classification layer to a frozen backbone."""

    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.backbone(x)
        h = h.flatten(start_dim=1)
        return self.fc(h)


# =====================================================================
# 4. LOSS FUNCTION
# =====================================================================
def infonce_loss(z1, z2, temperature=0.1):
    """Computes vectorized InfoNCE contrastive loss."""
    N = z1.shape[0]

    #L2 Normalize embeddings to unit vectors
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    #stack correlated views
    z = torch.cat([z1, z2], dim=0)  # [2N, D]

    #compute pairwise similarities
    sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]

    #mask out self-similarity (CPU instantiation to bypass DirectML graph bugs)
    mask = torch.eye(2 * N, device='cpu').to(z.device)
    sim = sim - (mask * 1e9)

    #define target labels (i matches with i + N)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels], dim=0)

    #categorical Cross Entropy Loss
    loss = F.cross_entropy(sim, labels)
    return loss


# =====================================================================
# 5. TRAINING LOOPS
# =====================================================================
def train_simclr(model, loader, optimizer, device, num_epochs):
    """Executes Self-Supervised pretraining using SimCLR framework."""
    model.train()

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}...")
        total_loss = 0
        batch_idx = 0

        #for each pair of augmented views, compute projections, calculate contrastive loss, and update model weights
        for aug1, aug2 in loader:
            aug1, aug2 = aug1.to(device), aug2.to(device)

            optimizer.zero_grad()
            _, z1 = model(aug1)
            _, z2 = model(aug2)
            loss = infonce_loss(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_idx += 1

            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {total_loss / len(loader):.4f}")

    return model


def train_fully_supervised(model, train_loader, test_loader, device, lr, num_epochs):
    """Trains an entire model (backbone + linear layer) from scratch using ground truth targets."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_losses": [], "train_accuracies": [],
        "test_losses": [], "test_accuracies": []
    }

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        #for each batch of standard images, run the model, compute cross-entropy loss against the correct labels, and update all network weights
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Evaluation Phase
        model.eval()
        total_test_loss = 0
        test_correct = 0
        test_total = 0

        #pass the validation/test images through the model without calculating gradients to monitor performance and accuracy on unseen data
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total

        # Save Metrics
        history["train_losses"].append(avg_train_loss)
        history["train_accuracies"].append(train_acc)
        history["test_losses"].append(avg_test_loss)
        history["test_accuracies"].append(test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print(
        f"\nFinal Supervised Target Accuracy: {history['test_accuracies'][-1]:.2f}%")
    return history


# =====================================================================
# 6. DOWNSTREAM EVALUATION & METRICS
# =====================================================================
def extract_backbone_features(backbone, images):
    """Helper utility extracting flattened raw representations directly from backbone."""
    h = backbone(images)
    return torch.flatten(h, start_dim=1)


def get_embeddings(model, loader, device, is_encoder_class=True):
    """Passes a dataset through the network to collect raw representations and labels."""
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        #extract and accumulate the raw feature vectors (embeddings) and true labels for downstream evaluation, skipping gradient calculations
        for images, lbls in loader:
            images = images.to(device)

            if is_encoder_class:
                h, _ = model(images)
            else:
                if hasattr(model, 'fc'):
                    backbone_extractor = nn.Sequential(
                        *list(model.children())[:-1])
                    h = backbone_extractor(images)
                else:
                    h = model(images)

            h = torch.flatten(h, start_dim=1)
            embeddings.append(h.cpu().numpy())
            labels.append(lbls.numpy())

    return np.concatenate(embeddings), np.concatenate(labels)


def evaluate_downstream(downstream_model, train_loader, test_loader, device, lr, num_epochs):
    """Trains a linear probe head on top of a completely frozen model backbone."""
    criterion = nn.CrossEntropyLoss()
    optimizer_eval = optim.Adam(downstream_model.fc.parameters(), lr=lr)

    history = {
        "train_losses": [], "train_accuracies": [],
        "test_losses": [], "test_accuracies": []
    }

    for epoch in range(num_epochs):
        #training Phase
        downstream_model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        #train only the linear classification head on top of the frozen backbone features using correct labels
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer_eval.zero_grad()
            outputs = downstream_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_eval.step()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        #evaluation Phase
        downstream_model.eval()
        total_test_loss = 0
        test_correct = 0
        test_total = 0

        #evaluate the linear head's classification performance on unseen test data without updating any weights
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = downstream_model(images)

                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total

        # Record metrics
        history["train_losses"].append(avg_train_loss)
        history["train_accuracies"].append(train_acc)
        history["test_losses"].append(avg_test_loss)
        history["test_accuracies"].append(test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print(
        f"\nFinal Accuracy on STL-10 Test Set: {history['test_accuracies'][-1]:.2f}%")
    return history


def knn_evaluate(train_embeddings, train_labels, test_embeddings, test_labels,
                 k=5, metric="cosine"):
    """Fits a non-parametric Scikit-Learn k-Nearest Neighbors classifier on raw embeddings."""
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(train_embeddings, train_labels)

    preds = knn.predict(test_embeddings)
    acc = accuracy_score(test_labels, preds)
    return acc, preds, knn


# =====================================================================
# 7. STATIC AND INTERACTIVE VISUALIZATIONS
# =====================================================================

def plot_evaluation_curves_interactive(history):
    """Generates an interactive twin subplot showing loss and accuracy curves side by side."""
    num_epochs = len(history['train_losses'])
    epochs = list(range(1, num_epochs + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Linear Classifier: Loss Progress</b>",
                        "<b>Linear Classifier: Accuracy Progress</b>")
    )

    # Loss Curves
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_losses'], mode='lines+markers',
                   name='Train Loss',
                   marker=dict(symbol='square', color='firebrick'),
                   line=dict(color='firebrick')), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=epochs, y=history['test_losses'], mode='lines+markers',
                   name='Test Loss',
                   marker=dict(symbol='circle', color='darkorange'),
                   line=dict(color='darkorange', dash='dash')), row=1, col=1)

    # Accuracy Curves
    fig.add_trace(go.Scatter(x=epochs, y=history['train_accuracies'],
                             mode='lines+markers', name='Train Acc',
                             marker=dict(symbol='square', color='forestgreen'),
                             line=dict(color='forestgreen')), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history['test_accuracies'],
                             mode='lines+markers', name='Test Acc',
                             marker=dict(symbol='circle', color='teal'),
                             line=dict(color='teal', dash='dash')), row=1,
                  col=2)

    fig.update_layout(title_text="Model Training Metrics Dashboard",
                      template="plotly_white", width=1100, height=500,
                      hovermode="x unified")
    fig.update_xaxes(title_text="Epoch", tickvals=epochs, row=1, col=1)
    fig.update_xaxes(title_text="Epoch", tickvals=epochs, row=1, col=2)

    max_loss = max(max(history['train_losses']), max(history['test_losses']))
    fig.update_yaxes(title_text="Loss", range=[0, max(2.0, max_loss * 1.1)],
                     row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 105], row=1, col=2)
    fig.show()


def plot_confusion_matrix_interactive(cm, class_names,
                                      title="k-NN Confusion Matrix"):
    """Plots interactive visual confusion matrices utilizing standard Plotly layout grids."""
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_stacked = df_cm.stack().reset_index()
    df_stacked.columns = ['True Class', 'Predicted Class', 'Count']

    fig = px.density_heatmap(
        df_stacked, x='Predicted Class', y='True Class', z='Count',
        text_auto=True,
        color_continuous_scale='Blues', title=f"<b>{title}</b>",
        labels={'Count': 'Predictions'}
    )
    fig.update_layout(xaxis_title="Predicted Class", yaxis_title="True Class",
                      width=750, height=700,
                      xaxis=dict(side='bottom'),
                      yaxis=dict(autorange='reversed'),
                      template="plotly_white")
    fig.show()


def plot_final_comparison_interactive(simclr_linear_acc, simclr_knn_acc,
                                      random_baseline_acc,
                                      supervised_baseline_acc):
    """Draws multi-category interactive Plotly bar metrics tracking experimental performance bounds."""
    data = {
        'Method': [
            '1. Random Features (Baseline)', '2. SimCLR + k-NN',
            '3. SimCLR + Linear Probe', '4. Fully Supervised (Upper Bound)'
        ],
        'Accuracy (%)': [random_baseline_acc, simclr_knn_acc,
                         simclr_linear_acc, supervised_baseline_acc],
        'Category': ['Baseline', 'Our SimCLR Model',
                     'Our SimCLR Model', 'Baseline']
    }
    df = pd.DataFrame(data)

    fig = px.bar(
        df, x='Method', y='Accuracy (%)', color='Category', text_auto='.2f',
        title="<b>Final Test Accuracy Comparison Across All Methods</b>",
        color_discrete_map={'Baseline': '#A0A0A0',
                            'Our SimCLR Model': '#1F77B4'}
    )
    fig.update_layout(xaxis_title="Evaluation Framework",
                      yaxis_title="Test Accuracy (%)",
                      yaxis=dict(range=[0, 105]), width=850, height=500,
                      template="plotly_white",
                      legend_title_text="Framework Type")
    fig.show()


def generate_and_plot_umap_interactive(model_or_backbone, loader, device,
                                       class_names, title,
                                       is_encoder_class=True):
    """Runs UMAP non-linear dimensionality reductions to map hyper-dimensional projections down to 2D."""
    model_or_backbone.eval()
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if is_encoder_class:
                h, _ = model_or_backbone(images)
            else:
                #if it's a standard torchvision ResNet, bypass the final fc layer
                if hasattr(model_or_backbone, 'fc'):
                    #extract features using everything except the last layer
                    backbone_extractor = nn.Sequential(
                        *list(model_or_backbone.children())[:-1])
                    h = backbone_extractor(images)
                else:
                    h = model_or_backbone(images)

            h = torch.flatten(h, start_dim=1)
            embeddings.append(h.cpu().numpy())
            labels_list.append(labels.numpy())

    all_embeddings = np.concatenate(embeddings, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(all_embeddings)

    label_strings = [class_names[idx] for idx in all_labels]
    df = pd.DataFrame({
        'UMAP 1': embedding_2d[:, 0], 'UMAP 2': embedding_2d[:, 1],
        'Class': label_strings
    })

    fig = px.scatter(
        df, x='UMAP 1', y='UMAP 2', color='Class', hover_name='Class',
        title=f"<b>{title}</b>",
        category_orders={"Class": class_names},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(width=850, height=600,
                      xaxis=dict(scaleanchor="y", scaleratio=1),
                      template="plotly_white")
    fig.show()



def visualize_knn_retrieval(knn, test_embeddings, test_set, train_labeled_set, class_names, num_examples=10):
    """Finds query image representations and charts their top 5 nearest neighbors."""
    class_query_indices = {}
    for idx, (_, label) in enumerate(test_set):
        if label not in class_query_indices:
            class_query_indices[label] = idx
        if len(class_query_indices) == num_examples:
            break

    fig, axes = plt.subplots(num_examples, 6, figsize=(14, 2.5 * num_examples))

    #STL-10 normalization/unnormalization constants
    mean = np.array([0.4408, 0.4279, 0.3867])
    std = np.array([0.2682, 0.2610, 0.2686])

    def unnormalize(tensor_img):
        # Move channel dimension to the end: (C, H, W) -> (H, W, C)
        img = tensor_img.permute(1, 2, 0).numpy()
        # Apply inverse normalization formula: (x * std) + mean
        img = (img * std) + mean
        return np.clip(img, 0, 1)

    for row_idx in range(num_examples):
        test_idx = class_query_indices[row_idx]
        query_vector = test_embeddings[test_idx].reshape(1, -1)
        query_img_tensor, query_label = test_set[test_idx]

        distances, neighbor_indices = knn.kneighbors(query_vector, n_neighbors=5)

        # Plot Source Query
        axes[row_idx, 0].imshow(unnormalize(query_img_tensor))
        axes[row_idx, 0].set_title(f"Query: {class_names[query_label]}",
                                   fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Plot Neighbors Space
        for col_idx, train_idx in enumerate(neighbor_indices[0]):
            train_img_tensor, train_label = train_labeled_set[train_idx]
            dist = distances[0][col_idx]
            color = "green" if train_label == query_label else "red"

            axes[row_idx, col_idx + 1].imshow(unnormalize(train_img_tensor))
            axes[row_idx, col_idx + 1].set_title(
                f"Match {col_idx + 1}: {class_names[train_label]}\n(Dist: {dist:.2f})",
                fontsize=9, color=color
            )
            axes[row_idx, col_idx + 1].axis('off')

    plt.tight_layout()
    plt.show()