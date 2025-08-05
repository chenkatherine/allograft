import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np

# FEATURE EXTRACTION ===
# Pulled from UNI documentation
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
import timm
from torchvision import transforms

local_dir = "assets/ckpts/uni2-h"
timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
model = timm.create_model(
    pretrained=False, **timm_kwargs
)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()
print("Loaded model")

# ROI feature extraction
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

# Get path to data
dataroot = "HE_TILES"

# Create image folder datasets for train/test and their data loaders
train_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'train'), transform=transform)
test_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'test'), transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# Extract patch features from the train and test datasets (returns dictionary of embeddings and labels)
train_features = extract_patch_features_from_dataloader(model, train_dataloader)
test_features = extract_patch_features_from_dataloader(model, test_dataloader)

# Convert these to torch
train_feats = torch.Tensor(train_features['embeddings'])
train_labels = torch.Tensor(train_features['labels']).type(torch.long)
test_feats = torch.Tensor(test_features['embeddings'])
test_labels = torch.Tensor(test_features['labels']).type(torch.long)


# MLP ===
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Constructor
# 2 layers, 2 classes
class MLPClassifier(nn.Module):
    """ Attempts to implement a multi-layer perceptron with two hidden layers using PyTorch. """
    def __init__(self, input_dim, hidden_dims=[1024, 512], num_classes=2, dropout=0.6):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), 
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers) # Combines all layers
    
    def forward(self, x):
        """ Forward pass """
        return self.classifier(x)


# Save model
import torch
import os
from pathlib import Path

def save_model_checkpoint(model, train_accs, test_accs, final_accuracy, save_dir="checkpoints"):
    """ Save the trained model and training history """
    Path(save_dir).mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_dim': model.classifier[0].in_features,
            'hidden_dims': [1024, 512],
            'num_classes': 2,
            'dropout': 0.6
        },
        'train_accuracies': train_accs,
        'test_accuracies': test_accs,
        'final_accuracy': final_accuracy,
        'training_completed': True
    }
    
    checkpoint_path = os.path.join(save_dir, 'mlp_classifier.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
    return checkpoint_path


def train_mlp(train_feats, train_labels, test_feats, test_labels, epochs=25, print_every=5):
    """ Takes training and test data and trains with 25 epochs. """
    model = MLPClassifier(train_feats.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_dataset = TensorDataset(train_feats.to(device), train_labels.to(device))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Store accuracies just in case
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_feats, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_feats)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            # Track training accuracy
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
        
        scheduler.step()
        
        # Calculate training accuracy for this epoch
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_feats.to(device))
            test_pred = test_outputs.argmax(dim=1)
            test_accuracy = (test_pred == test_labels.to(device)).float().mean().item()
            test_accuracies.append(test_accuracy)
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1:3d}/{epochs}] - "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Test Acc: {test_accuracy:.4f}")
    # Print final results
    final_test_accuracy = test_accuracies[-1]
    print(f"\nFinal Test Accuracy: {final_test_accuracy:.4f}")

    save_model_checkpoint(model, train_accuracies, test_accuracies, 
                          final_test_accuracy, save_dir="checkpoints")
    
    return final_test_accuracy, train_accuracies, test_accuracies, model


# AUROC ===
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn.functional as F

def plot_roc(model, test_feats, test_labels, device, save_path=None):
    """ Plots AUROC """
    model.eval()
    with torch.no_grad():
        outputs = model(test_feats.to(device))
        probabilities = F.softmax(outputs, dim=1)
        y_scores = probabilities[:, 1].cpu().numpy()
        y_true = test_labels.cpu().numpy()
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', lw=2, 
             label=f'MLP (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classifier performance')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    print(f"ROC-AUC: {roc_auc:.4f}")
    return roc_auc


def main():
    mlp_acc_raw, train_accs, test_accs, mlp_classifier = train_mlp(train_feats, train_labels, test_feats, test_labels)
    print(f"MLP (raw features): {mlp_acc_raw:.4f}")
    plot_roc(mlp_classifier, test_feats, test_labels, device, save_path='mlp_roc.png')

if __name__ == "__main__":
    main()
