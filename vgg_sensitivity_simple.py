"""
VGG-11 Gradient-Based Sensitivity Score Generation for CIFAR-10
Supports configurable metrics: Product, Magnitude, Gradient

Same methodology and output format as resnet_sensitivity_simple.py:
sensitivity_score = |gradient| × |weight| (Product)
sensitivity_score = |weight| (Magnitude)
sensitivity_score = |gradient| (Gradient)

The gradient is the mean of |∂L/∂w| over --num_batches batches. One CSV per
Conv2d/Linear weight (weight_sensitivity_scores_<name>.csv, columns index,
sensitivity_score), as read by VGG_multi_Strategy.py --score_dir.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import csv
import os
from tqdm import tqdm
import argparse

from core.models import vgg11_bn

# Configuration
BATCH_SIZE = 32
NUM_BATCHES = 50
DEFAULT_HELDOUT_SIZE = 5000
SEED = 42

MODELS = {'vgg11_bn': vgg11_bn}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate VGG-11 Sensitivity Scores (CIFAR-10)')
    parser.add_argument('--model', type=str, default='vgg11_bn', choices=sorted(MODELS),
                        help='Model architecture from core/models.py (default: vgg11_bn)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained VGG-11 checkpoint (.pt or .pth file), loaded with strict=True')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to download/store CIFAR-10 dataset (default: ./data)')
    parser.add_argument('--metric', type=str, default='product', choices=['product', 'magnitude', 'gradient'],
                        help='Sensitivity metric to use: product (|w|*|g|), magnitude (|w|), or gradient (|g|)')
    parser.add_argument('--output_dir', '--score_dir', dest='output_dir', type=str,
                        default='vgg_weight_sensitivity_score',
                        help='Directory to save sensitivity scores (--score_dir is an alias)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--num_batches', type=int, default=NUM_BATCHES,
                        help=f'Number of batches to accumulate gradients over (default: {NUM_BATCHES})')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda if available)')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed (default: {SEED})')
    parser.add_argument('--val_split', type=str, default='test', choices=['test', 'heldout'],
                        help='Images used for the gradients: test = CIFAR-10 test set (default); '
                             'heldout = the same seeded training split as VGG_multi_Strategy.py --val_split heldout')
    parser.add_argument('--heldout_size', type=int, default=DEFAULT_HELDOUT_SIZE,
                        help=f'Training images held out with --val_split heldout (default: {DEFAULT_HELDOUT_SIZE})')
    return parser.parse_args()

def set_seed(seed):
    """Seed all random number generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def heldout_split(num_train, heldout_size, seed):
    """Seeded split of training indices into (train, heldout); same as VGG_multi_Strategy.heldout_split"""
    if not 0 < heldout_size < num_train:
        raise ValueError(f"--heldout_size must be between 1 and {num_train - 1}, got {heldout_size}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_train, generator=generator).tolist()
    return perm[heldout_size:], perm[:heldout_size]

def load_model(model_name, checkpoint_path, device):
    """Load trained VGG-11 model for CIFAR-10 (10 classes)"""
    print("Loading VGG-11 model...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = MODELS[model_name](pretrained=False, device=device, num_classes=10)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    print(f"  ✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def load_data(data_path, batch_size, val_split, heldout_size, seed):
    """Load CIFAR-10 images for the gradient pass (test set by default)"""
    print("\nLoading CIFAR-10 data...")

    # Same evaluation transform as VGG_multi_Strategy.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if val_split == 'test':
        dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        _, heldout_idx = heldout_split(len(train_dataset), heldout_size, seed)
        dataset = Subset(train_dataset, heldout_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"  ✓ Loaded {len(dataset)} {val_split} samples")

    return loader

def compute_gradients(model, data_loader, num_batches, device):
    """Compute gradients over multiple batches"""
    print(f"\nComputing gradients over {num_batches} batches...")
    model.train()
    criterion = nn.CrossEntropyLoss()

    grad_accum = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    for batch_idx, (data, target) in enumerate(tqdm(data_loader, total=num_batches, desc="Computing gradients")):
        if batch_idx >= num_batches:
            break

        data, target = data.to(device), target.to(device)

        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum[name] += param.grad.abs()

    for name in grad_accum:
        grad_accum[name] /= num_batches

    print("  ✓ Gradient computation complete")
    return grad_accum

def save_sensitivity_scores(model, gradients, output_dir, metric='product'):
    """Compute and save sensitivity scores to CSV files"""
    print(f"\nSaving sensitivity scores to {output_dir}/ (Metric: {metric})...")
    os.makedirs(output_dir, exist_ok=True)

    total_layers = 0
    total_weights = 0
    total_non_finite = 0

    for name, param in tqdm(model.named_parameters(), desc="Saving scores"):
        if 'weight' not in name or param.dim() < 2:
            continue

        if name not in gradients:
            print(f"  ⚠ WARNING: No gradient for {name}, skipping...")
            continue

        grad = gradients[name].cpu().numpy().flatten()
        weight = param.data.cpu().numpy().flatten()

        # Compute sensitivity scores based on selected metric
        if metric == 'product':
            sensitivity_scores = np.abs(grad) * np.abs(weight)
        elif metric == 'magnitude':
            sensitivity_scores = np.abs(weight)
        elif metric == 'gradient':
            sensitivity_scores = np.abs(grad)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Non-finite scores are written as 0 (as in the other sensitivity scripts); report them
        non_finite = int(np.count_nonzero(~np.isfinite(sensitivity_scores)))
        if non_finite > 0:
            print(f"  ⚠ WARNING: {name}: {non_finite:,} non-finite scores (inf/nan) set to 0")
        total_non_finite += non_finite

        sensitivity_scores = np.nan_to_num(sensitivity_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Prepare all rows at once
        rows = [[idx, f'{score:.10e}'] for idx, score in enumerate(sensitivity_scores)]

        # Write all rows in one go
        csv_file = os.path.join(output_dir, f'weight_sensitivity_scores_{name}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'sensitivity_score'])
            writer.writerows(rows)

        total_layers += 1
        total_weights += len(sensitivity_scores)

        non_zero = np.count_nonzero(sensitivity_scores)
        print(f"  ✓ {name}: {len(sensitivity_scores):,} weights, "
              f"{non_zero:,} non-zero scores ({non_zero/len(sensitivity_scores)*100:.2f}%)")

    print(f"\n  ✓ Saved {total_layers} layers with {total_weights:,} total weights")
    if total_non_finite > 0:
        print(f"  ⚠ WARNING: {total_non_finite:,} non-finite scores in total were set to 0")
    print(f"  ✓ Sensitivity scores saved to {output_dir}/")

def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print(f"VGG-11 Sensitivity Score Generation (CIFAR-10)")
    print(f"Metric: {args.metric.upper()}")
    print(f"Output: {args.output_dir}")
    print(f"Batches: {args.num_batches} x {args.batch_size} | Seed: {args.seed} | Split: {args.val_split}")
    print("=" * 80)

    model = load_model(args.model, args.checkpoint, device)
    data_loader = load_data(args.data_path, args.batch_size, args.val_split, args.heldout_size, args.seed)
    gradients = compute_gradients(model, data_loader, args.num_batches, device)
    save_sensitivity_scores(model, gradients, args.output_dir, args.metric)

    print("\n" + "=" * 80)
    print("✓ Sensitivity score generation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Verify CSV files in: {args.output_dir}/")
    print(f"2. Run VGG_multi_Strategy.py --checkpoint {args.checkpoint} --score_dir {args.output_dir}")

if __name__ == "__main__":
    main()
