# -*- coding: utf-8 -*-
"""
PROPERLY CORRECTED DeiT + U-Net++ segmentation training script
- Fixed MSI integration at level 0
- Proper attention gate implementations
- Correct channel dimensions throughout
- Standardized with working UNet configuration
"""
import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# torchvision is only required for the non-albumentations augmentation path.
# Make it optional so importing this module for inference does not fail.
try:
    from torchvision import transforms
except Exception:
    transforms = None

import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================
# Configuration with Grid Search
# =====================
IMG_DIR = "/lfs/jsuri.isu/Sanchit_Segmentation/Hong Kong image Data/224x224-Images"
MASK_DIR = "/lfs/jsuri.isu/Sanchit_Segmentation/Hong Kong image Data/224x224-Masks"
IMG_SIZE = 224
NUM_WORKERS = 2
GRID_SEARCH_EPOCHS = 10  # Epochs for grid search
FINAL_EPOCHS = 200       # Epochs for final training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
BASE_RESULTS_DIR = '/lfs/jsuri.isu/San_Seg/Results'

# Grid Search Parameters
GRID_SEARCH_PARAMS = {
    'learning_rate': [1e-4, 5e-5, 1e-3, 5e-4],
    'batch_size': [8, 16, 32],
    'focal_alpha': [0.25, 0.5, 0.75],
    'focal_gamma': [1.0, 2.0, 3.0]
}

# Default values (will be overridden by grid search)
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

def print_parameters():
    print("Experiment Parameters:")
    print(f"  IMG_DIR: {IMG_DIR}")
    print(f"  MASK_DIR: {MASK_DIR}")
    print(f"  IMG_SIZE: {IMG_SIZE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  NUM_WORKERS: {NUM_WORKERS}")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  LR: {LR}")
    print(f"  DEVICE: {DEVICE}")
    print(f"  SEED: {SEED}")

# =====================
# Dataset (Same as UNet)
# =====================
class HKImageSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224, augment=True, use_albumentations=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.use_albumentations = use_albumentations
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p).replace('.jpg', '.png')) for p in self.img_paths]
        
        if use_albumentations:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            self.val_aug = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform_img = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.transform_mask = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
        
        if self.use_albumentations:
            if self.augment:
                augmented = self.aug(image=img, mask=mask)
            else:
                augmented = self.val_aug(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()
        else:
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            if self.augment and random.random() > 0.5:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                img = self.aug(img)
                random.seed(seed)
                mask = self.aug(mask)
            img = self.transform_img(img)
            mask = self.transform_mask(mask)
        
        mask = torch.clamp(mask, 0, 1)
        return img, mask

# =====================
# Model Components (Same as UNet)
# =====================
class MultiScaleInputBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, img_size=224):
        super().__init__()
        self.scales = [img_size, img_size//2, img_size//4]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ) for _ in self.scales
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(self.scales)*out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        feats = []
        target_size = self.scales[0]
        for i, s in enumerate(self.scales):
            xi = F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)
            feat = self.convs[i](xi)
            feat = F.interpolate(feat, size=(target_size, target_size), mode='bilinear', align_corners=False)
            feats.append(feat)
        x_cat = torch.cat(feats, dim=1)
        fused = self.fuse(x_cat)
        return fused

class DeiTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.deit = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
        self.patch_embed = self.deit.patch_embed
        self.pos_drop = self.deit.pos_drop
        self.blocks = self.deit.blocks
        self.norm = self.deit.norm
        self.num_layers = len(self.blocks)
        self.selected_layers = [2, 5, 8, 11]
        self.embed_dim = self.deit.embed_dim
        self.img_size = self.deit.patch_embed.img_size
        self.patch_size = self.deit.patch_embed.patch_size
        self.num_patches = self.deit.patch_embed.num_patches
        self.grid_size = self.deit.patch_embed.grid_size

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.deit.pos_embed[:, 1:, :])
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.selected_layers:
                feat = x
                feat_2d = feat.transpose(1,2).reshape(B, self.embed_dim, self.grid_size[0], self.grid_size[1])
                features.append(feat_2d)
        x = self.norm(x)
        return features

class FusionBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch1 + in_ch2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=2, padding=0)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, gating):
        batch_size, channels, height, width = x.size()
        
        theta_x = self.theta(x)
        phi_g = self.phi(gating)
        
        if phi_g.size(2) != theta_x.size(2) or phi_g.size(3) != theta_x.size(3):
            phi_g = F.interpolate(phi_g, size=theta_x.size()[2:], mode='bilinear', align_corners=False)
        
        f = F.relu(theta_x + phi_g)
        psi = self.psi(f)
        psi_up = F.interpolate(psi, size=(height, width), mode='bilinear', align_corners=False)
        y = psi_up * x
        y = self.final_conv(y)
        
        return y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv(x)

# =====================
# CORRECTED UNet++ Model
# =====================
class DeiTUNetPPCorrected(nn.Module):
    def __init__(self, num_classes=1, img_size=224):
        super().__init__()
        # Same encoder setup as UNet
        self.msi = MultiScaleInputBlock(in_ch=3, out_ch=64, img_size=img_size)
        self.encoder = DeiTEncoder(pretrained=True)
        self.msi_proj = nn.Conv2d(64, self.encoder.embed_dim, 1)
        self.fusions = nn.ModuleList([
            FusionBlock(self.encoder.embed_dim, self.encoder.embed_dim, self.encoder.embed_dim) for _ in range(4)
        ])
        
        embed_dim = self.encoder.embed_dim
        
        # UNet++ architecture with correct channel progression
        # We'll use a simpler channel progression: 64, 128, 256, 512
        
        # Level projections for encoder features  
        self.enc_proj1 = nn.Conv2d(embed_dim, 128, 1)  # Level 1
        self.enc_proj2 = nn.Conv2d(embed_dim, 256, 1)  # Level 2  
        self.enc_proj3 = nn.Conv2d(embed_dim, 512, 1)  # Level 3
        
        # UNet++ nested blocks
        # X_i_j where i is level (depth) and j is density index
        
        # Level 0 (MSI level - 224x224)
        self.conv0_0 = ConvBlock(64, 64)                    # MSI features
        self.conv0_1 = ConvBlock(64 + 64, 64)               # 0_0 + up from 1_0  
        self.conv0_2 = ConvBlock(64 + 64, 64)               # 0_1 + up from 1_1
        self.conv0_3 = ConvBlock(64 + 64, 64)               # 0_2 + up from 1_2
        
        # Level 1 (112x112)
        self.conv1_0 = ConvBlock(128, 128)                  # Encoder level 1
        self.conv1_1 = ConvBlock(128 + 128, 128)            # 1_0 + up from 2_0
        self.conv1_2 = ConvBlock(128 + 128, 128)            # 1_1 + up from 2_1
        
        # Level 2 (56x56) 
        self.conv2_0 = ConvBlock(256, 256)                  # Encoder level 2
        self.conv2_1 = ConvBlock(256 + 256, 256)            # 2_0 + up from 3_0
        
        # Level 3 (28x28) - Bottleneck
        self.conv3_0 = ConvBlock(512, 512)                  # Encoder level 3
        
        # Attention gates for skip connections - fix channel dimensions to match upsampled features
        self.att1_0 = AttentionBlock(128, 128, 64)          # For 1_0 with gating from up2_0 (128 channels)
        self.att1_1 = AttentionBlock(128, 128, 64)          # For 1_1 with gating from up2_1 (128 channels)
        self.att0_1 = AttentionBlock(64, 64, 32)            # For 0_0 with gating from up1_0 (64 channels)
        self.att0_2 = AttentionBlock(64, 64, 32)            # For 0_1 with gating from up1_1 (64 channels)
        self.att0_3 = AttentionBlock(64, 64, 32)            # For 0_2 with gating from up1_2 (64 channels)
        self.att2_0 = AttentionBlock(256, 256, 128)         # For 2_0 with gating from up3_0 (256 channels)
        
        # Upsampling layers
        self.up3_0 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(256, 128, 2, stride=2)  
        self.up2_1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1_0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # Final segmentation head 
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Multi-scale input processing
        msi_feat = self.msi(x)  # 64 channels, 224x224
        msi_proj = self.msi_proj(msi_feat)
        
        # DeiT encoder
        enc_feats = self.encoder(x)  # 4 features, all 14x14, 768 channels
        
        # Fuse with MSI
        fused_feats = [
            self.fusions[i](
                enc_feats[i],
                F.interpolate(msi_proj, size=enc_feats[i].shape[-2:], mode='bilinear', align_corners=False)
            )
            for i in range(4)
        ]
        
        # Project and upsample encoder features to correct scales and channels
        x1_enc = F.interpolate(self.enc_proj1(fused_feats[1]), size=(112, 112), mode='bilinear', align_corners=False)
        x2_enc = F.interpolate(self.enc_proj2(fused_feats[2]), size=(56, 56), mode='bilinear', align_corners=False)  
        x3_enc = F.interpolate(self.enc_proj3(fused_feats[3]), size=(28, 28), mode='bilinear', align_corners=False)
        
        # UNet++ forward pass
        # Initialize base nodes
        x0_0 = self.conv0_0(msi_feat)           # 64@224x224
        x1_0 = self.conv1_0(x1_enc)             # 128@112x112  
        x2_0 = self.conv2_0(x2_enc)             # 256@56x56
        x3_0 = self.conv3_0(x3_enc)             # 512@28x28
        
        # Dense connections with attention
        # Level 2 -> Level 1
        up3_0 = self.up3_0(x3_0)  # 256@56x56
        up3_0 = F.interpolate(up3_0, size=(56, 56), mode='bilinear', align_corners=False)
        att2_0 = self.att2_0(x2_0, up3_0)
        x2_1 = self.conv2_1(torch.cat([att2_0, up3_0], dim=1))  # 256@56x56
        
        # Level 1 connections  
        up2_0 = self.up2_0(x2_0)  # 128@112x112
        up2_0 = F.interpolate(up2_0, size=(112, 112), mode='bilinear', align_corners=False)
        att1_0 = self.att1_0(x1_0, up2_0)
        x1_1 = self.conv1_1(torch.cat([att1_0, up2_0], dim=1))  # 128@112x112
        
        up2_1 = self.up2_1(x2_1)  # 128@112x112  
        up2_1 = F.interpolate(up2_1, size=(112, 112), mode='bilinear', align_corners=False)
        att1_1 = self.att1_1(x1_1, up2_1)
        x1_2 = self.conv1_2(torch.cat([att1_1, up2_1], dim=1))  # 128@112x112
        
        # Level 0 connections
        up1_0 = self.up1_0(x1_0)  # 64@224x224
        up1_0 = F.interpolate(up1_0, size=(224, 224), mode='bilinear', align_corners=False)  
        att0_1 = self.att0_1(x0_0, up1_0)
        x0_1 = self.conv0_1(torch.cat([att0_1, up1_0], dim=1))  # 64@224x224
        
        up1_1 = self.up1_1(x1_1)  # 64@224x224
        up1_1 = F.interpolate(up1_1, size=(224, 224), mode='bilinear', align_corners=False)
        att0_2 = self.att0_2(x0_1, up1_1)  
        x0_2 = self.conv0_2(torch.cat([att0_2, up1_1], dim=1))  # 64@224x224
        
        up1_2 = self.up1_2(x1_2)  # 64@224x224
        up1_2 = F.interpolate(up1_2, size=(224, 224), mode='bilinear', align_corners=False)
        att0_3 = self.att0_3(x0_2, up1_2)
        x0_3 = self.conv0_3(torch.cat([att0_3, up1_2], dim=1))  # 64@224x224
        
        # Final output
        out = self.final(x0_3)
        out = torch.sigmoid(out)
        return out

# =====================
# Loss and Metrics (Same as UNet)
# =====================
def dice_loss(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def focal_dice_loss(pred, target, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    focal_loss = focal_loss.mean()
    dice_loss_val = dice_loss(pred, target)
    return 0.5 * focal_loss + 0.5 * dice_loss_val

def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return (2 * intersection / (union + 1e-8)).mean().item()

def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
    return (intersection / (union + 1e-8)).mean().item()

def precision_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    tp = (pred * target).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))
    return (tp / (tp + fp + 1e-8)).mean().item()

def recall_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    tp = (pred * target).sum(dim=(1,2,3))
    fn = ((1 - pred) * target).sum(dim=(1,2,3))
    return (tp / (tp + fn + 1e-8)).mean().item()

def roc_auc_score_segmentation(pred, target):
    """Calculate ROC AUC for segmentation masks"""
    try:
        # Flatten predictions and targets
        pred_flat = pred.cpu().numpy().flatten()
        target_flat = target.cpu().numpy().flatten()
        
        # Check if we have both positive and negative samples
        if len(np.unique(target_flat)) < 2:
            return 0.0  # Return 0 if only one class present
        
        # Calculate ROC AUC
        auc = roc_auc_score(target_flat, pred_flat)
        return auc
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC: {e}")
        return 0.0

# =====================
# Training & Evaluation
# =====================
def train_one_epoch(model, loader, optimizer, scheduler, epoch, focal_alpha=0.25, focal_gamma=2.0):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc=f"Train Epoch {epoch}", disable=True):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = focal_dice_loss(preds, masks, alpha=focal_alpha, gamma=focal_gamma)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    scheduler.step()
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def evaluate(model, loader, epoch, compute_loss=False, focal_alpha=0.25, focal_gamma=2.0):
    model.eval()
    dice, iou, prec, rec, roc_auc = 0, 0, 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Eval Epoch {epoch}", disable=True):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            if compute_loss:
                loss = focal_dice_loss(preds, masks, alpha=focal_alpha, gamma=focal_gamma)
                total_loss += loss.item() * imgs.size(0)
            dice += dice_score(preds, masks) * imgs.size(0)
            iou += iou_score(preds, masks) * imgs.size(0)
            prec += precision_score(preds, masks) * imgs.size(0)
            rec += recall_score(preds, masks) * imgs.size(0)
            roc_auc += roc_auc_score_segmentation(preds, masks) * imgs.size(0)
    n = len(loader.dataset)
    avg_loss = total_loss / n if compute_loss else None
    return (dice/n, iou/n, prec/n, rec/n, roc_auc/n, avg_loss) if compute_loss else (dice/n, iou/n, prec/n, rec/n, roc_auc/n)

# =====================
# Utility Functions
# =====================
def save_metrics(metrics, filepath):
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)

def save_metrics_json(metrics, filepath):
    import json
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def create_results_directory_structure(base_dir):
    dirs = {
        'plots': os.path.join(base_dir, 'plots'),
        'models': os.path.join(base_dir, 'models'),
        'metrics': {
            'base': os.path.join(base_dir, 'metrics'),
            'training': os.path.join(base_dir, 'metrics', 'training'),
            'validation': os.path.join(base_dir, 'metrics', 'validation')
        },
        'predictions': {
            'training': os.path.join(base_dir, 'predictions', 'training'),
            'validation': os.path.join(base_dir, 'predictions', 'validation')
        },
        'roc_data': os.path.join(base_dir, 'roc_data')
    }
    for dir_path in dirs.values():
        if isinstance(dir_path, dict):
            for subdir in dir_path.values():
                os.makedirs(subdir, exist_ok=True)
        else:
            os.makedirs(dir_path, exist_ok=True)
    return dirs

def print_epoch_header_custom():
    print("\n" + "="*110)
    print(f"{'EPOCH':<6} {'PHASE':<6} {'LOSS':<10} {'DICE':<10} {'IOU':<10} {'PREC':<10} {'REC':<10} {'ROC_AUC':<10} {'LR':<10}")
    print("="*110)

# =====================
# Grid Search Functions
# =====================
def train_with_params(train_loader, val_loader, params, epochs):
    """Train model with specific hyperparameters for given epochs"""
    # Create model
    model = DeiTUNetPPCorrected(num_classes=1, img_size=IMG_SIZE).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Setup optimizer and scheduler with current params
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_dice = 0
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, 
                                    params['focal_alpha'], params['focal_gamma'])
        
        # Validate
        val_dice, val_iou, val_prec, val_rec, val_roc_auc, val_loss = evaluate(
            model, val_loader, epoch, compute_loss=True, 
            focal_alpha=params['focal_alpha'], focal_gamma=params['focal_gamma'])
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
    
    return best_val_dice, model

def grid_search(train_loader, val_loader):
    """Perform grid search to find best hyperparameters"""
    print("\n" + "="*80)
    print("STARTING GRID SEARCH (10 epochs per combination)")
    print("="*80)
    
    import itertools
    
    # Generate all combinations
    param_names = list(GRID_SEARCH_PARAMS.keys())
    param_values = list(GRID_SEARCH_PARAMS.values())
    combinations = list(itertools.product(*param_values))
    
    best_score = 0
    best_params = None
    results = []
    
    total_combinations = len(combinations)
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    for i, combo in enumerate(combinations):
        # Create parameter dict
        params = dict(zip(param_names, combo))
        
        print(f"[{i+1}/{total_combinations}] Testing: LR={params['learning_rate']:.0e}, "
              f"BS={params['batch_size']}, α={params['focal_alpha']}, γ={params['focal_gamma']}")
        
        # Create data loaders with current batch size
        current_train_loader = DataLoader(train_loader.dataset, 
                                        batch_size=params['batch_size'], 
                                        shuffle=True, num_workers=NUM_WORKERS)
        current_val_loader = DataLoader(val_loader.dataset, 
                                      batch_size=params['batch_size'], 
                                      shuffle=False, num_workers=NUM_WORKERS)
        
        # Train with current params
        val_dice, _ = train_with_params(current_train_loader, current_val_loader, 
                                       params, GRID_SEARCH_EPOCHS)
        
        results.append({
            'params': params,
            'val_dice': val_dice
        })
        
        print(f"   → Validation Dice: {val_dice:.4f}")
        
        # Update best
        if val_dice > best_score:
            best_score = val_dice
            best_params = params
            print(f"   → NEW BEST! 🎉")
        
        print()
    
    print("="*80)
    print("GRID SEARCH COMPLETED")
    print(f"Best Validation Dice: {best_score:.4f}")
    print(f"Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Save grid search results
    grid_search_results = pd.DataFrame([
        {
            'learning_rate': r['params']['learning_rate'],
            'batch_size': r['params']['batch_size'], 
            'focal_alpha': r['params']['focal_alpha'],
            'focal_gamma': r['params']['focal_gamma'],
            'val_dice': r['val_dice']
        } for r in results
    ])
    
    return best_params, grid_search_results

# =====================
# Main with Grid Search
# =====================
def main():
    print_parameters()
    script_name = "deit_unetpp_corrected_gridsearch"
    results_dir = os.path.join(BASE_RESULTS_DIR, script_name)
    os.makedirs(results_dir, exist_ok=True)
    dirs = create_results_directory_structure(results_dir)
    
    # Dataset split (same as UNet)
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
    random.shuffle(all_imgs)
    n_total = len(all_imgs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_imgs = all_imgs[:n_train]
    val_imgs = all_imgs[n_train:n_train+n_val]
    test_imgs = all_imgs[n_train+n_val:]
    
    print(f"Dataset split: {n_train} train, {n_val} val, {n_test} test")
    
    # Prepare datasets
    train_set = HKImageSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, augment=True)
    val_set = HKImageSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, augment=False)
    test_set = HKImageSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, augment=False)
    train_set.img_paths = train_imgs
    train_set.mask_paths = [os.path.join(MASK_DIR, os.path.basename(p).replace('.jpg', '.png')) for p in train_imgs]
    val_set.img_paths = val_imgs
    val_set.mask_paths = [os.path.join(MASK_DIR, os.path.basename(p).replace('.jpg', '.png')) for p in val_imgs]
    test_set.img_paths = test_imgs
    test_set.mask_paths = [os.path.join(MASK_DIR, os.path.basename(p).replace('.jpg', '.png')) for p in test_imgs]
    
    # Initial data loaders for grid search (will be recreated with different batch sizes)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)
    
    # ===================
    # PHASE 1: GRID SEARCH
    # ===================
    best_params, grid_search_results = grid_search(train_loader, val_loader)
    
    # Save grid search results
    grid_search_results.to_csv(os.path.join(dirs['metrics']['base'], 'grid_search_results.csv'), index=False)
    print(f"\nGrid search results saved to: {os.path.join(dirs['metrics']['base'], 'grid_search_results.csv')}")
    
    # ===================
    # PHASE 2: FINAL TRAINING WITH BEST PARAMS
    # ===================
    print("\n" + "="*80)
    print(f"STARTING FINAL TRAINING WITH BEST PARAMETERS ({FINAL_EPOCHS} epochs)")
    print("="*80)
    print(f"Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Create final data loaders with best batch size
    final_train_loader = DataLoader(train_set, batch_size=best_params['batch_size'], 
                                   shuffle=True, num_workers=NUM_WORKERS)
    final_val_loader = DataLoader(val_set, batch_size=best_params['batch_size'], 
                                 shuffle=False, num_workers=NUM_WORKERS)
    final_test_loader = DataLoader(test_set, batch_size=best_params['batch_size'], 
                                  shuffle=False, num_workers=NUM_WORKERS)
    
    # Create final model
    final_model = DeiTUNetPPCorrected(num_classes=1, img_size=IMG_SIZE).to(DEVICE)
    num_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"\nTotal number of trainable parameters: {num_params:,}")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        final_model = nn.DataParallel(final_model)
    
    # Setup optimizer and scheduler with best parameters
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=FINAL_EPOCHS)
    
    # Training loop
    best_dice = 0
    best_epoch = 0
    best_val_metrics = None
    history = []
    
    print_epoch_header_custom()
    
    for epoch in range(1, FINAL_EPOCHS + 1):
        # Training
        train_loss = train_one_epoch(final_model, final_train_loader, final_optimizer, final_scheduler, epoch,
                                    best_params['focal_alpha'], best_params['focal_gamma'])
        train_dice, train_iou, train_prec, train_rec, train_roc_auc = evaluate(
            final_model, final_train_loader, epoch, compute_loss=False,
            focal_alpha=best_params['focal_alpha'], focal_gamma=best_params['focal_gamma'])
        
        # Validation
        val_dice, val_iou, val_prec, val_rec, val_roc_auc, val_loss = evaluate(
            final_model, final_val_loader, epoch, compute_loss=True,
            focal_alpha=best_params['focal_alpha'], focal_gamma=best_params['focal_gamma'])
        
        lr = final_optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"{epoch:<6} {'TRAIN':<6} {train_loss:<10.4f} {train_dice:<10.4f} {train_iou:<10.4f} {train_prec:<10.4f} {train_rec:<10.4f} {train_roc_auc:<10.4f} {lr:<10.2e}")
        print(f"{epoch:<6} {'VAL':<6} {val_loss:<10.4f} {val_dice:<10.4f} {val_iou:<10.4f} {val_prec:<10.4f} {val_rec:<10.4f} {val_roc_auc:<10.4f} {'N/A':<10}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_val_metrics = {
                'epoch': epoch,
                'dice': val_dice,
                'iou': val_iou,
                'precision': val_prec,
                'recall': val_rec,
                'roc_auc': val_roc_auc,
                'best_params': best_params
            }
            
            # Save best model
            torch.save(final_model.state_dict(), os.path.join(dirs['models'], 'best_deit_unetpp_corrected_final.pth'))
            print(f"\n🎉 NEW BEST MODEL! Epoch {epoch}, Dice: {val_dice:.4f}")
            
            # Save best metrics
            save_metrics(best_val_metrics, os.path.join(dirs['metrics']['base'], 'best_final_model_metrics.csv'))
            save_metrics_json(best_val_metrics, os.path.join(dirs['metrics']['base'], 'best_final_model_metrics.json'))
        
        # Track history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_roc_auc': train_roc_auc,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_roc_auc': val_roc_auc,
            'lr': lr
        })
    
    # ===================
    # FINAL EVALUATION
    # ===================
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model for final evaluation
    final_model.load_state_dict(torch.load(os.path.join(dirs['models'], 'best_deit_unetpp_corrected_final.pth')))
    
    # Test evaluation
    test_dice, test_iou, test_prec, test_rec, test_roc_auc = evaluate(
        final_model, final_test_loader, "FINAL_TEST", compute_loss=False,
        focal_alpha=best_params['focal_alpha'], focal_gamma=best_params['focal_gamma'])
    
    # Final results
    final_results = {
        'grid_search_results': 'Saved in grid_search_results.csv',
        'best_hyperparameters': best_params,
        'best_val_dice': best_dice,
        'best_val_epoch': best_epoch,
        'final_test_metrics': {
            'dice': test_dice,
            'iou': test_iou,
            'precision': test_prec,
            'recall': test_rec,
            'roc_auc': test_roc_auc
        }
    }
    
    # Save all results
    pd.DataFrame(history).to_csv(os.path.join(dirs['metrics']['base'], 'final_training_history.csv'), index=False)
    save_metrics_json(final_results, os.path.join(dirs['metrics']['base'], 'final_results_summary.json'))
    
    # Print summary
    print(f"\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📊 Grid Search: Tested {len(grid_search_results)} parameter combinations")
    print(f"🏆 Best Validation Dice: {best_dice:.4f} (Epoch {best_epoch})")
    print(f"🧪 Final Test Results:")
    print(f"   • Dice Score: {test_dice:.4f}")
    print(f"   • IoU Score: {test_iou:.4f}")
    print(f"   • Precision: {test_prec:.4f}")
    print(f"   • Recall: {test_rec:.4f}")
    print(f"   • ROC AUC: {test_roc_auc:.4f}")
    print(f"📁 Results saved in: {results_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
