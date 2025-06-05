# Import libraries and packages
import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np

from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the get_encoder function to load the UNI2-h model
from huggingface_hub import login

# login with user access token

# downloading weights and creating model
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
# model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, init_values=1e-5, dynamic_img_size=False)
# transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
# model.eval()
# model.to(device)
# transform

# pretrained=True needed to load UNI2-h weights (and download weights for the first time)
timm_kwargs = {
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
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

# Use the get_encoder function which properly sets up all the required parameters
# model, transform = get_encoder(enc_name='uni2-h', device=device)

# ROI feature extraction
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

# get path to example data
dataroot = "HE_IMAGES"

# also set the image max to none or else it throws a DecompressionBomb Error
Image.MAX_IMAGE_PIXELS = None

# create some image folder datasets for train/test and their data laoders
train_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'train'), transform=transform)
test_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'test'), transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# extract patch features from the train and test datasets (returns dictionary of embeddings and labels)
train_features = extract_patch_features_from_dataloader(model, train_dataloader)
test_features = extract_patch_features_from_dataloader(model, test_dataloader)

# convert these to torch
train_feats = torch.Tensor(train_features['embeddings'])
train_labels = torch.Tensor(train_features['labels']).type(torch.long)
test_feats = torch.Tensor(test_features['embeddings'])
test_labels = torch.Tensor(test_features['labels']).type(torch.long)

# ROI few-shot evaluation (based on ProtoNet)
from uni.downstream.eval_patch_features.fewshot import eval_fewshot

fewshot_episodes, fewshot_dump = eval_fewshot(
    train_feats = train_feats,
    train_labels = train_labels,
    test_feats = test_feats,
    test_labels = test_labels,
    n_iter = 10000, # draw 500 few-shot episodes
    n_way = 2, # use all class examples; originally 2
    n_shot = 10, # 4 examples per class (as we don't have that many)
    n_query = test_feats.shape[0], # evaluate on all test samples
    center_feats = False, # True
    normalize_feats = False, # True
    average_feats = False, # True
)

# how well it does
print(fewshot_episodes)

# summary
print(fewshot_dump)
print()


# linear probe
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe

linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
    train_feats = train_feats,
    train_labels = train_labels,
    valid_feats = None ,
    valid_labels = None,
    test_feats = test_feats,
    test_labels = test_labels,
    max_iter = 5000,
    verbose= True,
)
print("Linear probe evaluation")
print_metrics(linprobe_eval_metrics)
print()


print("MLP")
# Import statements
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Constructor
# 2 layers, 2 classes
class MLPClassifier(nn.Module):
    """
    Attempts to implement a multi-layer perceptron with two hidden layers using PyTorch. 
    Implements batch normalization, dropout regulation, and learning rate scheduling
    to protect against overfitting.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=2, dropout=0.7):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim), # Creates fully connected layer
                nn.BatchNorm1d(hidden_dim), # Normalizes inputs for training stability
                nn.ReLU(), # ReLU activation function
                nn.Dropout(dropout) # Dropout rate = 0.5
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers) # Combines all layers
    
    def forward(self, x):
        """ Implements the forward pass method. """
        return self.classifier(x)

def train_mlp(train_feats, train_labels, test_feats, test_labels, epochs=200, print_every=10):
    """ Takes training and test data and trains with 200 epochs. """
    model = MLPClassifier(train_feats.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_dataset = TensorDataset(train_feats.to(device), train_labels.to(device))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Store accuracies for plotting/analysis if needed
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
    
    return final_test_accuracy, train_accuracies, test_accuracies

# Function call
mlp_acc_raw, train_accs, test_accs = train_mlp(train_feats, train_labels, test_feats, test_labels)
print(f"MLP (raw features): {mlp_acc_raw:.4f}")
