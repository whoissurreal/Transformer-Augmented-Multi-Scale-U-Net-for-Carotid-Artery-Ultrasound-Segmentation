"""
DeiT + Attention U-Net Segmentation Script

- Encoder: DeiT-base (pretrained, multi-scale input, feature fusion)
- Decoder: Attention U-Net style with attention gates (from C5-att-UNet.py)
- Attention Mechanism: Shallow attention gates between encoder and decoder features
- Dataset: Loads from Hong Kong image Data/224x224-Images and 224x224-Masks
- Training: Dice + BCE loss, Adam optimizer, cosine annealing
- Evaluation: Dice, IoU, Precision, Recall
- Visualization: Input vs. predicted mask
- TorchScript export supported

Attention Mechanism:
- Theta path: Downsamples encoder features
- Phi path: Processes decoder features  
- Psi path: Generates attention weights via sigmoid
- Applied attention weights to encoder features before skip connections

Author: [Your Name]
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

# torchvision is only needed for optional training-time transforms/visualization.
# Make it optional so inference can run even if local torchvision is incompatible.
try:
    from torchvision import transforms, utils
    from torchvision.utils import make_grid
except Exception:  # ImportError, AttributeError, etc.
    transforms = None
    utils = None
    make_grid = None

import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob
from sklearn.metrics import roc_curve, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================
# Configuration
# =====================
IMG_DIR = '/lfs/jsuri.isu/Sanchit_Segmentation/Augmented/224x224-Images-Augmented'
MASK_DIR = '/lfs/jsuri.isu/Sanchit_Segmentation/Augmented/224x224-Masks-Augmented'
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 200
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
# Hardcoded base results directory
BASE_RESULTS_DIR = '/lfs/jsuri.isu/Sanchit_Segmentation/results'

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
    print(f"  BASE_RESULTS_DIR: {BASE_RESULTS_DIR}")

# =====================
# Dataset
# =====================
class HKImageSegmentationDataset(Dataset):
    """
    Dataset for Hong Kong image Data segmentation task.
    Assumes images are in IMG_DIR and masks in MASK_DIR with matching filenames.
    Supports albumentations for robust augmentation.
    """
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
            self.aug = transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
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
            mask = augmented['mask'].unsqueeze(0).float()  # add channel dim
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
        # More gentle binarization - keep some gradient information
        mask = torch.clamp(mask, 0, 1)
        return img, mask

# =====================
# Multiscale Input Block
# =====================
class MultiScaleInputBlock(nn.Module):
    """
    Processes input at multiple scales with shallow CNNs, then fuses features.
    """
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
            # Always interpolate, even if shape matches
            feat = F.interpolate(feat, size=(target_size, target_size), mode='bilinear', align_corners=False)
            feats.append(feat)
        x_cat = torch.cat(feats, dim=1)
        fused = self.fuse(x_cat)
        return fused

# =====================
# DeiT Encoder Wrapper
# =====================
class DeiTEncoder(nn.Module):
    """
    DeiT encoder that extracts multi-scale features from intermediate transformer layers.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.deit = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
        self.patch_embed = self.deit.patch_embed
        self.pos_drop = self.deit.pos_drop
        self.blocks = self.deit.blocks
        self.norm = self.deit.norm
        self.num_layers = len(self.blocks)
        self.selected_layers = [2, 5, 8, 11]  # 0-based: layers 3,6,9,12
        self.embed_dim = self.deit.embed_dim
        self.img_size = self.deit.patch_embed.img_size
        self.patch_size = self.deit.patch_embed.patch_size
        self.num_patches = self.deit.patch_embed.num_patches
        self.grid_size = self.deit.patch_embed.grid_size

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        # Remove class token from pos_embed to match patch embeddings
        x = self.pos_drop(x + self.deit.pos_embed[:, 1:, :])
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.selected_layers:
                feat = x
                feat_2d = feat.transpose(1,2).reshape(B, self.embed_dim, self.grid_size[0], self.grid_size[1])
                features.append(feat_2d)
        x = self.norm(x)
        return features  # List of 4 feature maps

# =====================
# Fusion Block
# =====================
class FusionBlock(nn.Module):
    """
    Fuses transformer and multiscale features via concatenation and 1x1 conv.
    """
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

# =====================
# Attention Block (from C5-att-UNet.py)
# =====================
class AttentionBlock(nn.Module):
    """
    Shallow attention gate from encoder feature `x` + decoder `gating`.
    PyTorch implementation of the attention mechanism from C5-att-UNet.py
    """
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        
        # Theta path (encoder features) - downsample
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=2, padding=0)
        
        # Phi path (decoder features) - no downsampling
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        
        # Psi path (attention weights)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        # Final 1x1 conv to restore channel dimensions
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, gating):
        # x: encoder features, gating: decoder features
        batch_size, channels, height, width = x.size()
        
        # Theta path - downsample encoder features
        theta_x = self.theta(x)  # (B, inter_channels, H/2, W/2)
        
        # Phi path - process decoder features
        phi_g = self.phi(gating)  # (B, inter_channels, H_g, W_g)
        
        # Resize phi_g to match theta_x spatial dimensions
        if phi_g.size(2) != theta_x.size(2) or phi_g.size(3) != theta_x.size(3):
            phi_g = F.interpolate(phi_g, size=theta_x.size()[2:], mode='bilinear', align_corners=False)
        
        # Add and apply ReLU
        f = F.relu(theta_x + phi_g)
        
        # Generate attention weights
        psi = self.psi(f)  # (B, 1, H/2, W/2)
        
        # Upsample attention weights to original spatial size
        psi_up = F.interpolate(psi, size=(height, width), mode='bilinear', align_corners=False)
        
        # Apply attention weights to encoder features
        y = psi_up * x
        
        # Final 1x1 conv
        y = self.final_conv(y)
        
        return y

# =====================
# U-Net Decoder Block with Attention
# =====================
class DecoderBlockWithAttention(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, gating_ch=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        
        # Attention gate
        if gating_ch is None:
            gating_ch = out_ch
        self.attention = AttentionBlock(skip_ch, gating_ch, out_ch)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        # Always interpolate, even if shape matches
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        # Apply attention to skip connection
        attended_skip = self.attention(skip, x)
        
        # Concatenate upsampled features with attended skip connection
        x = torch.cat([x, attended_skip], dim=1)
        x = self.conv(x)
        return x

# =====================
# U-Net Decoder Block (Original - keep for compatibility)
# =====================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x, skip):
        x = self.up(x)
        # Always interpolate, even if shape matches
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# =====================
# Full Model
# =====================
class DeiTAttentionUNet(nn.Module):
    """
    DeiT + Attention U-Net segmentation model with multiscale input and fusion blocks.
    Incorporates attention gates from C5-att-UNet.py for better feature selection.
    """
    def __init__(self, num_classes=1, img_size=224):
        super().__init__()
        self.msi = MultiScaleInputBlock(in_ch=3, out_ch=64, img_size=img_size)
        self.encoder = DeiTEncoder(pretrained=True)
        # Project multiscale features to match transformer feature dims
        self.msi_proj = nn.Conv2d(64, self.encoder.embed_dim, 1)
        # Fusion blocks for each skip connection
        self.fusions = nn.ModuleList([
            FusionBlock(self.encoder.embed_dim, self.encoder.embed_dim, self.encoder.embed_dim) for _ in range(4)
        ])
        # Decoder
        self.decoder4 = DecoderBlockWithAttention(self.encoder.embed_dim, self.encoder.embed_dim, 256)
        self.decoder3 = DecoderBlockWithAttention(256, self.encoder.embed_dim, 128)
        self.decoder2 = DecoderBlockWithAttention(128, self.encoder.embed_dim, 64)
        self.decoder1 = DecoderBlockWithAttention(64, 64, 32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        msi_feat = self.msi(x)
        msi_proj = self.msi_proj(msi_feat)
        enc_feats = self.encoder(x)
        # Fuse multiscale features with transformer features
        fused_feats = [
            self.fusions[i](
                enc_feats[i],
                F.interpolate(msi_proj, size=enc_feats[i].shape[-2:], mode='bilinear', align_corners=False)
            )
            for i in range(4)
        ]
        # Decoder with attention gates (following Attention U-Net pattern)
        # Level 4: deepest features (bottleneck)
        d4 = self.decoder4(fused_feats[3], fused_feats[2])  # Attention on encoder level 2
        
        # Level 3: attention on encoder level 1
        d3 = self.decoder3(d4, fused_feats[1])
        
        # Level 2: attention on encoder level 0  
        d2 = self.decoder2(d3, fused_feats[0])
        
        # Level 1: attention on multiscale input features
        d1 = self.decoder1(d2, msi_feat)
        
        out = self.seg_head(d1)
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        out = torch.sigmoid(out)
        return out

# =====================
# Loss Functions
# =====================
def dice_loss(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def focal_dice_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss combined with Dice loss for better handling of class imbalance.
    This is the primary loss function for segmentation training.
    """
    # Focal loss component
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    focal_loss = focal_loss.mean()
    
    # Dice loss component
    dice_loss_val = dice_loss(pred, target)
    
    return 0.5 * focal_loss + 0.5 * dice_loss_val

def get_loss_components(pred, target):
    """
    Get individual loss components for debugging.
    """
    bce = F.binary_cross_entropy(pred, target, reduction='mean')
    dsc = dice_loss(pred, target)
    focal_dice = focal_dice_loss(pred, target)
    return {
        'bce': bce.item(),
        'dice': dsc.item(),
        'focal_dice': focal_dice.item()
    }

# =====================
# Metrics
# =====================
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

# =====================
# Training & Evaluation
# =====================
def train_one_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc=f"Train Epoch {epoch}", disable=True):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = focal_dice_loss(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    scheduler.step()
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def evaluate(model, loader, epoch, compute_loss=False):
    model.eval()
    dice, iou, prec, rec = 0, 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Eval Epoch {epoch}", disable=True):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            if compute_loss:
                loss = focal_dice_loss(preds, masks)
                total_loss += loss.item() * imgs.size(0)
            dice += dice_score(preds, masks) * imgs.size(0)
            iou += iou_score(preds, masks) * imgs.size(0)
            prec += precision_score(preds, masks) * imgs.size(0)
            rec += recall_score(preds, masks) * imgs.size(0)
    n = len(loader.dataset)
    avg_loss = total_loss / n if compute_loss else None
    return (dice/n, iou/n, prec/n, rec/n, avg_loss) if compute_loss else (dice/n, iou/n, prec/n, rec/n)

# =====================
# Visualization
# =====================
def visualize_batch(imgs, masks, preds=None, n=4):
    imgs = imgs[:n].cpu().numpy().transpose(0,2,3,1)
    masks = masks[:n].cpu().numpy().squeeze(1)
    if preds is not None:
        preds = (preds[:n].cpu().numpy().squeeze(1) > 0.5).astype(np.float32)
    fig, axs = plt.subplots(n, 3 if preds is not None else 2, figsize=(10, 3*n))
    for i in range(n):
        axs[i,0].imshow((imgs[i]*[0.229,0.224,0.225]+[0.485,0.456,0.406]).clip(0,1))
        axs[i,0].set_title('Image')
        axs[i,1].imshow(masks[i], cmap='gray')
        axs[i,1].set_title('Mask')
        if preds is not None:
            axs[i,2].imshow(preds[i], cmap='gray')
            axs[i,2].set_title('Pred')
        for j in range(axs.shape[1]):
            axs[i,j].axis('off')
    plt.tight_layout()
    plt.show()

def create_results_directory_structure(base_dir):
    """Create the directory structure for results"""
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

def save_metrics(metrics, filepath):
    """Save metrics to CSV file"""
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)

def save_predictions(predictions, labels, probabilities, filepath_base):
    """Save predictions and probabilities to both CSV and NPZ files"""
    df = pd.DataFrame({
        'true_labels': labels,
        'predictions': predictions,
        'probabilities': probabilities
    })
    df.to_csv(f"{filepath_base}.csv", index=False)
    np.savez(
        f"{filepath_base}.npz",
        predictions=predictions,
        labels=labels,
        probabilities=probabilities
    )

def save_roc_data(fpr, tpr, thresholds, filepath):
    """Save ROC curve data to CSV"""
    df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    df.to_csv(filepath, index=False)

def print_directory_structure(base_dir):
    """Print the directory structure of results"""
    print("\nResults Directory Structure:")
    print(f"Base Directory: {os.path.abspath(base_dir)}")
    print("\nGenerated Files:")
    categories = {
        "Metrics": ["metrics/training/*.csv", "metrics/validation/*.csv", "metrics/*.csv"],
        "Predictions": ["predictions/training/*.csv", "predictions/validation/*.csv"],
        "Models": ["models/*.pth"],
        "Plots": ["plots/*.png"],
        "ROC Data": ["roc_data/*.csv"]
    }
    for category, patterns in categories.items():
        print(f"\n{category}:")
        for pattern in patterns: 
            files = glob.glob(os.path.join(base_dir, pattern))
            for file in files:
                print(f"  - {os.path.relpath(file, base_dir)}")

def save_side_by_side_images(imgs, masks, preds, save_dir, prefix="val_epoch"):
    """Save 10 images with real image and predicted mask side by side."""
    import torchvision.utils as vutils
    os.makedirs(save_dir, exist_ok=True)
    imgs = imgs[:10].cpu()
    masks = masks[:10].cpu()
    preds = (preds[:10].cpu() > 0.5).float()
    for i in range(imgs.size(0)):
        img = imgs[i]
        mask = masks[i]
        pred = preds[i]
        # Denormalize image
        img_np = img.numpy().transpose(1,2,0)
        img_np = (img_np * [0.229,0.224,0.225]) + [0.485,0.456,0.406]
        img_np = (img_np * 255).clip(0,255).astype('uint8')
        mask_np = (mask.squeeze().numpy() * 255).astype('uint8')
        pred_np = (pred.squeeze().numpy() * 255).astype('uint8')
        # Stack real image, GT mask, and predicted mask horizontally
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img_np)
        mask_pil = PILImage.fromarray(mask_np).convert('RGB')
        pred_pil = PILImage.fromarray(pred_np).convert('RGB')
        combined = PILImage.new('RGB', (img_pil.width*3, img_pil.height))
        combined.paste(img_pil, (0,0))
        combined.paste(mask_pil, (img_pil.width,0))
        combined.paste(pred_pil, (img_pil.width*2,0))
        combined.save(os.path.join(save_dir, f"{prefix}_sample_{i+1}.png"))

def save_metrics_json(metrics, filepath):
    import json
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def print_epoch_header_custom():
    print("\n" + "="*100)
    print(f"{'EPOCH':<6} {'PHASE':<6} {'LOSS':<10} {'DICE':<10} {'IOU':<10} {'PREC':<10} {'REC':<10} {'ROC_AUC':<10} {'LR':<10}")
    print("="*100)

# =====================
# Main
# =====================
# Add a new hardcoded directory for test samples
TEST_SAMPLES_DIR = '/lfs/jsuri.isu/Sanchit_Segmentation/results/deit_attention_unet_segmentation/test_samples'

# Add a function to save test images and masks

def save_test_images_and_masks(imgs, masks, preds, save_dir, filenames):
    """Save side-by-side images and predicted masks for the test set."""
    import torchvision.utils as vutils
    os.makedirs(save_dir, exist_ok=True)
    imgs = imgs.cpu()
    masks = masks.cpu()
    preds = (preds.cpu() > 0.5).float()
    for i in range(imgs.size(0)):
        img = imgs[i]
        mask = masks[i]
        pred = preds[i]
        # Denormalize image
        img_np = img.numpy().transpose(1,2,0)
        img_np = (img_np * [0.229,0.224,0.225]) + [0.485,0.456,0.406]
        img_np = (img_np * 255).clip(0,255).astype('uint8')
        mask_np = (mask.squeeze().numpy() * 255).astype('uint8')
        pred_np = (pred.squeeze().numpy() * 255).astype('uint8')
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img_np)
        mask_pil = PILImage.fromarray(mask_np).convert('RGB')
        pred_pil = PILImage.fromarray(pred_np).convert('RGB')
        combined = PILImage.new('RGB', (img_pil.width*3, img_pil.height))
        combined.paste(img_pil, (0,0))
        combined.paste(mask_pil, (img_pil.width,0))
        combined.paste(pred_pil, (img_pil.width*2,0))
        combined.save(os.path.join(save_dir, f"{os.path.splitext(filenames[i])[0]}_sidebyside.png"))
        # Save predicted mask as PNG
        pred_pil.save(os.path.join(save_dir, f"{os.path.splitext(filenames[i])[0]}_pred.png"))


def main():
    # Print parameters
    print_parameters()
    # Results directory setup
    script_name = "deit_attention_unet_segmentation"  # Updated name to reflect attention mechanism
    results_dir = os.path.join(BASE_RESULTS_DIR, script_name)
    os.makedirs(results_dir, exist_ok=True)
    dirs = create_results_directory_structure(results_dir)

    # Dataset split (80-10-10)
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
    random.shuffle(all_imgs)
    n_total = len(all_imgs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_imgs = all_imgs[:n_train]
    val_imgs = all_imgs[n_train:n_train+n_val]
    test_imgs = all_imgs[n_train+n_val:]
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
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = DeiTAttentionUNet(num_classes=1, img_size=IMG_SIZE).to(DEVICE)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params}")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dice = 0
    history = []  # To store all epoch metrics
    best_epoch = 0
    best_val_metrics = None
    best_roc_data = None
    best_preds = None
    best_labels = None
    best_probs = None
    best_imgs = None
    best_masks = None
    best_pred_imgs = None
    # Create folder for best samples
    best_samples_dir = os.path.join(dirs['plots'], 'best_samples')
    os.makedirs(best_samples_dir, exist_ok=True)
    print_epoch_header_custom()
    for epoch in range(1, EPOCHS+1):
        # --- Training ---
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        # Full evaluation on training set
        train_dice, train_iou, train_prec, train_rec = evaluate(model, train_loader, epoch)
        all_train_preds, all_train_labels, all_train_probs = [], [], []
        with torch.no_grad():
            for timgs, tmasks in train_loader:
                timgs = timgs.to(DEVICE)
                tmasks = tmasks.to(DEVICE)
                tpreds = model(timgs)
                all_train_preds.append((tpreds > 0.5).cpu().numpy().astype(np.uint8))
                all_train_labels.append(tmasks.cpu().numpy().astype(np.uint8))
                all_train_probs.append(tpreds.cpu().numpy())
        all_train_preds_np = np.concatenate(all_train_preds).reshape(-1)
        all_train_labels_np = np.concatenate(all_train_labels).reshape(-1)
        all_train_probs_np = np.concatenate(all_train_probs).reshape(-1)
        try:
            train_roc_auc = roc_auc_score(all_train_labels_np, all_train_probs_np)
        except Exception:
            train_roc_auc = float('nan')
        # --- Validation ---
        val_dice, val_iou, val_prec, val_rec, val_loss = evaluate(model, val_loader, epoch, compute_loss=True)
        all_val_preds, all_val_labels, all_val_probs = [], [], []
        with torch.no_grad():
            for vimgs, vmasks in val_loader:
                vimgs = vimgs.to(DEVICE)
                vmasks = vmasks.to(DEVICE)
                vpreds = model(vimgs)
                all_val_preds.append((vpreds > 0.5).cpu().numpy().astype(np.uint8))
                all_val_labels.append(vmasks.cpu().numpy().astype(np.uint8))
                all_val_probs.append(vpreds.cpu().numpy())
        all_val_preds_np = np.concatenate(all_val_preds).reshape(-1)
        all_val_labels_np = np.concatenate(all_val_labels).reshape(-1)
        all_val_probs_np = np.concatenate(all_val_probs).reshape(-1)
        try:
            val_roc_auc = roc_auc_score(all_val_labels_np, all_val_probs_np)
        except Exception:
            val_roc_auc = float('nan')
        lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else LR
        # Print table
        print(f"{epoch:<6} {'TRAIN':<6} {train_loss:<10.4f} {train_dice:<10.4f} {train_iou:<10.4f} {train_prec:<10.4f} {train_rec:<10.4f} {train_roc_auc:<10.4f} {lr:<10.2e}")
        print(f"{epoch:<6} {'VAL':<6} {val_loss:<10.4f} {val_dice:<10.4f} {val_iou:<10.4f} {val_prec:<10.4f} {val_rec:<10.4f} {val_roc_auc:<10.4f} {'N/A':<10}")
        # Calculate ROC curve data for validation set
        fpr, tpr, thresholds = roc_curve(all_val_labels_np, all_val_probs_np)
        # Save best model and associated data
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_val_metrics = {'epoch': epoch, 'dice': val_dice, 'iou': val_iou, 'precision': val_prec, 'recall': val_rec, 'roc_auc': val_roc_auc}
            best_roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
            best_preds = all_val_preds_np.tolist()
            best_labels = all_val_labels_np.tolist()
            best_probs = all_val_probs_np.tolist()
            # Save best model
            torch.save(model.state_dict(), os.path.join(dirs['models'], 'best_deit_attention_unet.pth'))
            print(f"[Main] Saved best model at epoch {epoch} with Dice {val_dice:.4f}")
            # Save best metrics as CSV and JSON
            save_metrics(best_val_metrics, os.path.join(dirs['metrics']['base'], 'best_model_metrics.csv'))
            save_metrics_json(best_val_metrics, os.path.join(dirs['metrics']['base'], 'best_model_metrics.json'))
            # Save best ROC data as CSV and JSON
            save_roc_data(fpr, tpr, thresholds, os.path.join(dirs['roc_data'], 'best_model_roc.csv'))
            save_metrics_json(best_roc_data, os.path.join(dirs['roc_data'], 'best_model_roc.json'))
            # Save best predictions as CSV and NPZ
            save_predictions(all_val_preds_np, all_val_labels_np, all_val_probs_np, os.path.join(dirs['predictions']['validation'], 'best_model'))
            # Get a batch from the validation loader for visualization
            imgs, masks = next(iter(val_loader))
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.no_grad():
                preds = model(imgs)
            # Save 10 best segmented images in a dedicated folder
            save_side_by_side_images(imgs, masks, preds, best_samples_dir, prefix=f"best_epoch_{epoch}")
        # Track all epoch metrics
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_roc_auc': val_roc_auc
        })
    # Save all epoch metrics as CSV
    pd.DataFrame(history).to_csv(os.path.join(dirs['metrics']['base'], 'all_epochs_metrics.csv'), index=False)
    # Export TorchScript
    model.eval()
    model_to_trace = model.module if isinstance(model, nn.DataParallel) else model
    example = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
    traced = torch.jit.trace(model_to_trace, example)
    traced.save(os.path.join(dirs['models'], 'deit_attention_unet_scripted.pt'))
    print("[Main] TorchScript model exported as deit_attention_unet_scripted.pt")
    # Save summary JSON
    summary = {
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'best_val_metrics': best_val_metrics,
        'training_config': {
            'IMG_DIR': IMG_DIR,
            'MASK_DIR': MASK_DIR,
            'IMG_SIZE': IMG_SIZE,
            'BATCH_SIZE': BATCH_SIZE,
            'NUM_WORKERS': NUM_WORKERS,
            'EPOCHS': EPOCHS,
            'LR': LR,
            'DEVICE': DEVICE,
            'SEED': SEED
        }
    }
    save_metrics_json(summary, os.path.join(dirs['metrics']['base'], 'training_summary.json'))
    print_directory_structure(results_dir)

    # =====================
    # Test set evaluation on best model
    # =====================
    print("\n[Main] Evaluating best model on test set...")
    # Reload best model
    model_to_test = DeiTAttentionUNet(num_classes=1, img_size=IMG_SIZE).to(DEVICE)
    state_dict = torch.load(os.path.join(dirs['models'], 'best_deit_attention_unet.pth'), map_location=DEVICE)
    # If the keys are all prefixed with 'module.', remove it
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    model_to_test.load_state_dict(state_dict)
    model_to_test.eval()
    # Evaluate on test set
    test_dice, test_iou, test_prec, test_rec = evaluate(model_to_test, test_loader, 0)
    all_test_preds, all_test_labels, all_test_probs, all_test_imgs, all_test_masks, all_test_filenames = [], [], [], [], [], []
    with torch.no_grad():
        for timgs, tmasks in test_loader:
            timgs = timgs.to(DEVICE)
            tmasks = tmasks.to(DEVICE)
            tpreds = model_to_test(timgs)
            all_test_preds.append((tpreds > 0.5).cpu().numpy().astype(np.uint8))
            all_test_labels.append(tmasks.cpu().numpy().astype(np.uint8))
            all_test_probs.append(tpreds.cpu().numpy())
            all_test_imgs.append(timgs.cpu())
            all_test_masks.append(tmasks.cpu())
    all_test_preds_np = np.concatenate(all_test_preds).reshape(-1)
    all_test_labels_np = np.concatenate(all_test_labels).reshape(-1)
    all_test_probs_np = np.concatenate(all_test_probs).reshape(-1)
    try:
        test_roc_auc = roc_auc_score(all_test_labels_np, all_test_probs_np)
    except Exception:
        test_roc_auc = float('nan')
    # Save test metrics
    test_metrics = {
        'dice': test_dice,
        'iou': test_iou,
        'precision': test_prec,
        'recall': test_rec,
        'roc_auc': test_roc_auc
    }
    save_metrics(test_metrics, os.path.join(dirs['metrics']['base'], 'test_metrics.csv'))
    save_metrics_json(test_metrics, os.path.join(dirs['metrics']['base'], 'test_metrics.json'))
    # Save test ROC data
    fpr, tpr, thresholds = roc_curve(all_test_labels_np, all_test_probs_np)
    save_roc_data(fpr, tpr, thresholds, os.path.join(dirs['roc_data'], 'test_roc.csv'))
    save_metrics_json({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}, os.path.join(dirs['roc_data'], 'test_roc.json'))
    # Save test predictions as CSV and NPZ
    save_predictions(all_test_preds_np, all_test_labels_np, all_test_probs_np, os.path.join(dirs['predictions']['validation'], 'test_model'))
    # Save side-by-side images and predicted masks for the test set
    # Flatten all_test_imgs and all_test_masks
    all_test_imgs_tensor = torch.cat(all_test_imgs, dim=0)
    all_test_masks_tensor = torch.cat(all_test_masks, dim=0)
    all_test_preds_tensor = torch.from_numpy(np.concatenate(all_test_preds, axis=0))
    # Use test_set.img_paths for filenames
    test_filenames = [os.path.basename(p) for p in test_set.img_paths]
    save_test_images_and_masks(all_test_imgs_tensor, all_test_masks_tensor, all_test_preds_tensor, TEST_SAMPLES_DIR, test_filenames)
    print(f"[Main] Test images and masks saved to {TEST_SAMPLES_DIR}")

if __name__ == '__main__':
    main() 