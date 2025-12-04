# -*- coding: utf-8 -*-
"""
PROPERLY CORRECTED DeiT + UNet3+ segmentation training script  
- Fixed spatial resolutions in full-scale connections
- Proper attention gate implementations throughout
- True UNet3+ architecture with all levels connected
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
    'learning_rate': [5e-5, 5e-4],
    'batch_size': [8],
    'focal_alpha': [0.5, 0.75],
    'focal_gamma': [3.0]
}

# Default values (will be overridden by grid search)
BATCH_SIZE = 8
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
# CORRECTED UNet3+ Model  
# =====================
class DeiTUNet3PlusCorrected(nn.Module):
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
        
        # UNet3+ uses 5 levels: level 4 (deepest) to level 0 (output)
        # Each level gets 320 channels = 64*5 (from 5 inputs)
        out_ch = 64  # Base channel count per input
        
        # Encoder projections - convert transformer features to base channels
        self.enc_proj0 = nn.Conv2d(embed_dim, out_ch, 1)  # 768 -> 64
        self.enc_proj1 = nn.Conv2d(embed_dim, out_ch, 1)  
        self.enc_proj2 = nn.Conv2d(embed_dim, out_ch, 1)
        self.enc_proj3 = nn.Conv2d(embed_dim, out_ch, 1)
        
        # UNet3+ Decoder blocks - fix channel dimensions
        self.decoder3 = ConvBlock(4 * out_ch, out_ch)  # Level 3: gets 4 encoder inputs (4*64=256) -> 64 channels
        self.decoder2 = ConvBlock(4 * out_ch, out_ch)  # Level 2: gets 4 inputs (4*64=256) -> 64 channels 
        self.decoder1 = ConvBlock(4 * out_ch, out_ch)  # Level 1: gets 4 inputs (4*64=256) -> 64 channels
        self.decoder0 = ConvBlock(4 * out_ch, out_ch)  # Level 0: gets 4 inputs (4*64=256) -> 64 channels
        
        # Attention gates for each connection - fix channel dimensions
        # Use smaller gating channel counts to match actual tensor sizes
        # Level 3 attention gates (4 encoder inputs)
        self.att3_e0 = AttentionBlock(out_ch, out_ch, out_ch//2)  
        self.att3_e1 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att3_e2 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att3_e3 = AttentionBlock(out_ch, out_ch, out_ch//2)
        
        # Level 2 attention gates (4 encoder + 1 decoder input)
        self.att2_e0 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att2_e1 = AttentionBlock(out_ch, out_ch, out_ch//2)  
        self.att2_e2 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att2_d3 = AttentionBlock(out_ch, out_ch, out_ch//2)
        
        # Level 1 attention gates (3 encoder + 2 decoder inputs)  
        self.att1_e0 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att1_e1 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att1_d3 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att1_d2 = AttentionBlock(out_ch, out_ch, out_ch//2)
        
        # Level 0 attention gates (2 encoder + 3 decoder inputs)
        self.att0_e0 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att0_d3 = AttentionBlock(out_ch, out_ch, out_ch//2)
        self.att0_d2 = AttentionBlock(out_ch, out_ch, out_ch//2) 
        self.att0_d1 = AttentionBlock(out_ch, out_ch, out_ch//2)
        
        # Upsampling layers for decoder features - match new decoder output dimensions
        self.up_d3_to_d2 = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)  # 28x28 -> 56x56
        self.up_d3_to_d1 = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=4)  # 28x28 -> 112x112
        self.up_d3_to_d0 = nn.ConvTranspose2d(out_ch, out_ch, 8, stride=8)  # 28x28 -> 224x224
        
        self.up_d2_to_d1 = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)  # 56x56 -> 112x112  
        self.up_d2_to_d0 = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=4)  # 56x56 -> 224x224
        
        self.up_d1_to_d0 = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)  # 112x112 -> 224x224
        
        # Final segmentation head - match new decoder output dimensions
        self.final = nn.Conv2d(out_ch, num_classes, 1)

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
        
        # Project encoder features and scale to appropriate resolutions
        # UNet3+ typically has 5 levels: 224, 112, 56, 28, 14
        e0 = F.interpolate(self.enc_proj0(fused_feats[0]), size=(224, 224), mode='bilinear', align_corners=False)  # MSI level
        e1 = F.interpolate(self.enc_proj1(fused_feats[1]), size=(112, 112), mode='bilinear', align_corners=False)
        e2 = F.interpolate(self.enc_proj2(fused_feats[2]), size=(56, 56), mode='bilinear', align_corners=False) 
        e3 = F.interpolate(self.enc_proj3(fused_feats[3]), size=(28, 28), mode='bilinear', align_corners=False)  # Bottleneck
        
        # UNet3+ Decoder with full-scale connections and attention
        
        # === Decoder Level 3 (28x28) ===
        # Gets inputs from all encoder levels (e0, e1, e2, e3) - downscaled to 28x28
        d3_e0 = F.adaptive_avg_pool2d(e0, (28, 28))  # 224->28
        d3_e1 = F.adaptive_avg_pool2d(e1, (28, 28))  # 112->28  
        d3_e2 = F.adaptive_avg_pool2d(e2, (28, 28))  # 56->28
        d3_e3 = e3  # Already 28x28
        
        # Apply attention gates using the deepest feature as gating signal
        gating_d3 = d3_e3  # Use deepest feature as gating signal (64 channels)
        d3_inputs = torch.cat([
            self.att3_e0(d3_e0, gating_d3),
            self.att3_e1(d3_e1, gating_d3), 
            self.att3_e2(d3_e2, gating_d3),
            self.att3_e3(d3_e3, gating_d3)
        ], dim=1)
        d3 = self.decoder3(d3_inputs)  # 64 channels, 28x28
        
        # === Decoder Level 2 (56x56) ===  
        # Gets inputs from e0,e1,e2 + upsampled d3
        d2_e0 = F.adaptive_avg_pool2d(e0, (56, 56))  # 224->56
        d2_e1 = F.adaptive_avg_pool2d(e1, (56, 56))  # 112->56
        d2_e2 = e2  # Already 56x56
        d2_d3 = F.interpolate(self.up_d3_to_d2(d3), size=(56, 56), mode='bilinear', align_corners=False)
        
        gating_d2 = d2_d3  # Use decoder feature as gating signal
        d2_inputs = torch.cat([
            self.att2_e0(d2_e0, gating_d2),
            self.att2_e1(d2_e1, gating_d2),
            self.att2_e2(d2_e2, gating_d2), 
            self.att2_d3(d2_d3, gating_d2)
        ], dim=1)
        d2 = self.decoder2(d2_inputs)  # 64 channels, 56x56
        
        # === Decoder Level 1 (112x112) ===
        # Gets inputs from e0,e1 + upsampled d3,d2
        d1_e0 = F.adaptive_avg_pool2d(e0, (112, 112))  # 224->112  
        d1_e1 = e1  # Already 112x112
        d1_d3 = F.interpolate(self.up_d3_to_d1(d3), size=(112, 112), mode='bilinear', align_corners=False)
        d1_d2 = F.interpolate(self.up_d2_to_d1(d2), size=(112, 112), mode='bilinear', align_corners=False)
        
        gating_d1 = d1_d3  # Use decoder feature as gating signal
        d1_inputs = torch.cat([
            self.att1_e0(d1_e0, gating_d1),
            self.att1_e1(d1_e1, gating_d1),
            self.att1_d3(d1_d3, gating_d1),
            self.att1_d2(d1_d2, gating_d1)
        ], dim=1)
        d1 = self.decoder1(d1_inputs)  # 64 channels, 112x112
        
        # === Decoder Level 0 (224x224) === 
        # Gets inputs from e0 + upsampled d3,d2,d1
        d0_e0 = e0  # Already 224x224
        d0_d3 = F.interpolate(self.up_d3_to_d0(d3), size=(224, 224), mode='bilinear', align_corners=False)
        d0_d2 = F.interpolate(self.up_d2_to_d0(d2), size=(224, 224), mode='bilinear', align_corners=False)
        d0_d1 = F.interpolate(self.up_d1_to_d0(d1), size=(224, 224), mode='bilinear', align_corners=False)
        
        gating_d0 = d0_d3  # Use decoder feature as gating signal
        d0_inputs = torch.cat([
            self.att0_e0(d0_e0, gating_d0),
            self.att0_d3(d0_d3, gating_d0),
            self.att0_d2(d0_d2, gating_d0), 
            self.att0_d1(d0_d1, gating_d0)
        ], dim=1)
        d0 = self.decoder0(d0_inputs)  # 64 channels, 224x224
        
        # Final output
        out = self.final(d0)
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
# Grid Search Functions
# =====================
def train_with_params(train_loader, val_loader, params, epochs):
    """Train model with specific hyperparameters for given epochs"""
    # Create model
    model = DeiTUNet3PlusCorrected(num_classes=1, img_size=IMG_SIZE).to(DEVICE)
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
# Main with Grid Search
# =====================
def main():
    print("\n" + "="*80)
    print("DeiT UNet3+ with Grid Search Hyperparameter Optimization")
    print("="*80)
    print(f"Grid Search: {GRID_SEARCH_EPOCHS} epochs per combination")
    print(f"Final Training: {FINAL_EPOCHS} epochs with best parameters")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    script_name = "deit_unet3p_grid_search"
    results_dir = os.path.join(BASE_RESULTS_DIR, script_name)
    os.makedirs(results_dir, exist_ok=True)
    dirs = create_results_directory_structure(results_dir)
    
    # Dataset split
    print("\nPreparing datasets...")
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
    random.shuffle(all_imgs)
    n_total = len(all_imgs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_imgs = all_imgs[:n_train]
    val_imgs = all_imgs[n_train:n_train+n_val]
    test_imgs = all_imgs[n_train+n_val:]
    
    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Prepare datasets (using default batch size for grid search)
    train_set = HKImageSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, augment=True)
    val_set = HKImageSegmentationDataset(IMG_DIR, MASK_DIR, IMG_SIZE, augment=False)
    train_set.img_paths = train_imgs
    train_set.mask_paths = [os.path.join(MASK_DIR, os.path.basename(p).replace('.jpg', '.png')) for p in train_imgs]
    val_set.img_paths = val_imgs
    val_set.mask_paths = [os.path.join(MASK_DIR, os.path.basename(p).replace('.jpg', '.png')) for p in val_imgs]
    
    # Initial data loaders for grid search
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)
    
    # STEP 1: Grid Search (10 epochs)
    print("\n" + "="*80)
    print("STEP 1: GRID SEARCH PHASE")
    print("="*80)
    best_params, grid_results = grid_search(train_loader, val_loader)
    
    # Save grid search results
    grid_results.to_csv(os.path.join(dirs['metrics']['base'], 'grid_search_results.csv'), index=False)
    
    # STEP 2: Final Training with Best Parameters (200 epochs)
    print("\n" + "="*80)
    print("STEP 2: FINAL TRAINING PHASE (200 epochs)")
    print("="*80)
    print(f"Training with best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Create final data loaders with best batch size
    final_train_loader = DataLoader(train_set, batch_size=best_params['batch_size'], 
                                   shuffle=True, num_workers=NUM_WORKERS)
    final_val_loader = DataLoader(val_set, batch_size=best_params['batch_size'], 
                                 shuffle=False, num_workers=NUM_WORKERS)
    
    # Create final model
    final_model = DeiTUNet3PlusCorrected(num_classes=1, img_size=IMG_SIZE).to(DEVICE)
    num_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        final_model = nn.DataParallel(final_model)
    
    # Setup optimizer and scheduler with best parameters
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
    
    # Final training loop
    best_dice = 0
    history = []
    best_epoch = 0
    best_val_metrics = None
    
    print_epoch_header_custom()
    for epoch in range(1, FINAL_EPOCHS + 1):
        # Training
        train_loss = train_one_epoch(final_model, final_train_loader, optimizer, scheduler, epoch,
                                    best_params['focal_alpha'], best_params['focal_gamma'])
        train_dice, train_iou, train_prec, train_rec, train_roc_auc = evaluate(
            final_model, final_train_loader, epoch, focal_alpha=best_params['focal_alpha'], 
            focal_gamma=best_params['focal_gamma'])
        
        # Validation
        val_dice, val_iou, val_prec, val_rec, val_roc_auc, val_loss = evaluate(
            final_model, final_val_loader, epoch, compute_loss=True,
            focal_alpha=best_params['focal_alpha'], focal_gamma=best_params['focal_gamma'])
        
        lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"{epoch:<6} {'TRAIN':<6} {train_loss:<10.4f} {train_dice:<10.4f} {train_iou:<10.4f} {train_prec:<10.4f} {train_rec:<10.4f} {train_roc_auc:<10.4f} {lr:<10.2e}")
        print(f"{epoch:<6} {'VAL':<6} {val_loss:<10.4f} {val_dice:<10.4f} {val_iou:<10.4f} {val_prec:<10.4f} {val_rec:<10.4f} {val_roc_auc:<10.4f} {'N/A':<10}")
        
        # Save best model (only from the 200-epoch training)
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_val_metrics = {
                'epoch': epoch, 'dice': val_dice, 'iou': val_iou, 
                'precision': val_prec, 'recall': val_rec, 'roc_auc': val_roc_auc,
                'best_params': best_params
            }
            torch.save(final_model.state_dict(), os.path.join(dirs['models'], 'best_deit_unet3p_grid_search.pth'))
            print(f"[Main] Saved best model at epoch {epoch} with Dice {val_dice:.4f}")
            
            # Save metrics
            save_metrics(best_val_metrics, os.path.join(dirs['metrics']['base'], 'best_model_metrics.csv'))
            save_metrics_json(best_val_metrics, os.path.join(dirs['metrics']['base'], 'best_model_metrics.json'))
        
        # Track history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss, 
            'train_dice': train_dice,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_roc_auc': val_roc_auc,
            'learning_rate': lr
        })
    
    # Save all training metrics
    pd.DataFrame(history).to_csv(os.path.join(dirs['metrics']['base'], 'final_training_metrics.csv'), index=False)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Validation Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"\nBest Hyperparameters Used:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nModel saved as: best_deit_unet3p_grid_search.pth")
    print(f"Results directory: {results_dir}")
    print("="*80)

if __name__ == '__main__':
    main()