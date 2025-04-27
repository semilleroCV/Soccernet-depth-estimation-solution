import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
import numpy as np
import random
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

import sys
sys.path.append('/ibex/user/perezpnf/SoccernetChallenge/')
sys.path.append('../')

from model import RelativeDepthAnything
# Import custom loss functions instead of CombinedLoss
from loss import ScaleAndShiftInvariantLoss, TrimmedProcrustesLoss, loss_dict
from train_utils import (
    set_seed, save_checkpoint, load_checkpoint, 
    get_learning_rate, log_metrics, log_images, 
    visualize_depth, AverageMeter
)
from data.soccernet_mde import get_dataloader

from metrics import (
    compute_mse, compute_rmse, compute_abs_rel, compute_sq_rel,
    compute_log10, compute_rmse_log, compute_threshold,
    compute_scale_and_shift, compute_metrics, MetricTracker
)

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Depth Anything V2 for Relative Depth Estimation')
    parser.add_argument('--data_dir', type=str, default=Config.data_dir, 
                        help='Path to the dataset directory')
    parser.add_argument('--crop_size', type=int, default=Config.crop_size, 
                        help='Size for random crop during training')
    parser.add_argument('--sport', type=str, default=Config.sport, 
                        choices=['foot', 'basket'], help='Sport to train on')
    
    parser.add_argument('--encoder', type=str, default=Config.encoder, 
                        choices=['vits', 'vitb', 'vitl', 'vitg'], help='Model encoder to use')
    parser.add_argument('--pretrained_weights', type=str, default=Config.pretrained_weights, 
                        help='Path to pretrained weights')
    
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--learning_rate', type=float, default=Config.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=Config.weight_decay)
    
    parser.add_argument('--checkpoint_dir', type=str, default=Config.checkpoint_dir)
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=Config.seed, help='Random seed')
    
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--log_freq', type=int, default=Config.log_freq, help='Logging frequency')
    
    # Add arguments for optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], 
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup', 
                        choices=['cosine_warmup', 'cosine', 'step', 'none'], 
                        help='Learning rate scheduler to use')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of warmup epochs for scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--lr_decay_epochs', nargs='+', type=int, default=[30, 60, 90], 
                        help='Epochs at which to decay learning rate for step scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help='Learning rate decay factor for step scheduler')
    
    # Add arguments for validation frequency
    parser.add_argument('--val_freq', type=int, default=1, 
                        help='Validation frequency in epochs')
    parser.add_argument('--save_ckpt_freq', type=int, default=5, 
                        help='Checkpoint saving frequency in epochs')
    
    # Add loss function arguments
    parser.add_argument('--loss_type', type=str, default='ssimse', 
                        choices=['ssimse', 'ssimae'], help='Loss function to use')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Alpha parameter for loss functions (gradient regularization weight)')
    parser.add_argument('--scales', type=int, default=4, 
                        help='Number of scales for gradient loss')
    parser.add_argument('--reduction', type=str, default='batch-based', 
                        choices=['batch-based', 'image-based'], help='Reduction method for loss')
    
    # Add arguments for data augmentation control
    parser.add_argument('--disable_horizontal_flip', action='store_true', 
                        help='Disable horizontal flipping augmentation')
    parser.add_argument('--disable_color_jitter', action='store_true',
                        help='Disable color jittering augmentation')
    parser.add_argument('--disable_rotation', action='store_true',
                        help='Disable rotation augmentation')
    
    args = parser.parse_args()
    return args


def get_optimizer(args, model):
    """Get optimizer based on args"""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optimizer


def get_scheduler(args, optimizer):
    """Get learning rate scheduler based on args"""
    if args.scheduler == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from transformers import get_cosine_schedule_with_warmup
        
        # Calculate number of training steps
        num_training_steps = args.epochs * (args.train_steps_per_epoch or 100)
        num_warmup_steps = args.warmup_epochs * (args.train_steps_per_epoch or 100)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate
        )
    else:
        scheduler = None
    
    return scheduler


def get_loss_function(args):
    """Get loss function based on args"""
    if args.loss_type == 'ssimse':
        criterion = ScaleAndShiftInvariantLoss(
            alpha=args.alpha, 
            scales=args.scales, 
            reduction=args.reduction
        )
    elif args.loss_type == 'ssimae':
        criterion = TrimmedProcrustesLoss(
            alpha=args.alpha, 
            scales=args.scales, 
            reduction=args.reduction
        )
    else:
        # Fallback to using the loss dictionary
        losses = loss_dict()
        criterion = losses.get(args.loss_type, losses['ssimse'])
    
    return criterion


def ddp_setup():
    """Set up distributed training"""
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    return rank, world_size


def create_comparison_image(rgb, depth_gt, depth_pred, error_map=None):
    """
    Create a side-by-side comparison image with RGB, ground truth depth, predicted depth,
    and optionally an error map.
    
    Args:
        rgb (torch.Tensor): RGB image tensor (C, H, W)
        depth_gt (torch.Tensor): Ground truth depth tensor (H, W)
        depth_pred (torch.Tensor): Predicted depth tensor (H, W)
        error_map (torch.Tensor, optional): Error map tensor (H, W)
        
    Returns:
        PIL.Image: Comparison image
    """
    # Convert tensors to numpy arrays
    rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
    depth_gt_np = depth_gt.cpu().numpy()
    depth_pred_np = depth_pred.detach().cpu().numpy()
    
    # Normalize RGB to [0, 1]
    rgb_np = np.clip(rgb_np, 0, 1)
    
    # Create figure with subplots
    if error_map is not None:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot RGB image
    axs[0].imshow(rgb_np)
    axs[0].set_title('RGB Image')
    axs[0].axis('off')
    
    # Plot ground truth depth
    vmin, vmax = np.nanpercentile(depth_gt_np[depth_gt_np > 0], (1, 99))
    im_gt = axs[1].imshow(depth_gt_np, cmap='plasma', vmin=vmin, vmax=vmax)
    axs[1].set_title('Ground Truth Depth')
    axs[1].axis('off')
    plt.colorbar(im_gt, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Plot predicted depth
    im_pred = axs[2].imshow(depth_pred_np, cmap='plasma', vmin=vmin, vmax=vmax)
    axs[2].set_title('Predicted Depth')
    axs[2].axis('off')
    plt.colorbar(im_pred, ax=axs[2], fraction=0.046, pad=0.04)
    
    # Plot error map if provided
    if error_map is not None:
        error_np = error_map.cpu().numpy()
        vmin_err, vmax_err = np.nanpercentile(error_np[depth_gt_np > 0], (1, 99))
        im_err = axs[3].imshow(error_np, cmap='inferno', vmin=vmin_err, vmax=vmax_err)
        axs[3].set_title('Error Map')
        axs[3].axis('off')
        plt.colorbar(im_err, ax=axs[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Convert matplotlib figure to PIL Image
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_argb())
    plt.close(fig)
    
    return img


def log_images_to_wandb(rgb, depth_gt, depth_pred, phase, sample_idx, step):
    """
    Log images to wandb with enhanced visualization.
    
    Args:
        rgb (torch.Tensor): RGB image tensor
        depth_gt (torch.Tensor): Ground truth depth tensor
        depth_pred (torch.Tensor): Predicted depth tensor
        phase (str): 'train' or 'val'
        sample_idx (int): Sample index
        step (int): Training step
    """
    # Convert RGB to numpy for wandb
    rgb_img = rgb.cpu()
    
    # Visualize depth maps
    depth_gt_vis = visualize_depth(depth_gt)
    depth_pred_vis = visualize_depth(depth_pred)
    
    # Create error map
    valid_mask = (depth_gt > 0) & torch.isfinite(depth_gt)
    error_map = torch.zeros_like(depth_gt)
    if valid_mask.any():
        # Compute absolute relative error
        abs_rel_error = torch.abs(depth_pred - depth_gt) / (depth_gt + 1e-10)
        # Apply mask
        error_map[valid_mask] = abs_rel_error[valid_mask].detach()
    
    # Create comparison image
    comparison_img = create_comparison_image(rgb, depth_gt, depth_pred, error_map)
    
    # Log to wandb
    wandb.log({
        f'{phase}/rgb_{sample_idx}': wandb.Image(rgb_img.permute(1, 2, 0).numpy()),
        f'{phase}/depth_gt_{sample_idx}': wandb.Image(depth_gt_vis),
        f'{phase}/depth_pred_{sample_idx}': wandb.Image(depth_pred_vis),
        f'{phase}/depth_error_{sample_idx}': wandb.Image(visualize_depth(error_map)),
        f'{phase}/comparison_{sample_idx}': wandb.Image(comparison_img)
    }, step=step)


def validate_one_epoch(model, val_loader, criterion, device, epoch, args, rank, world_size):
    """Validate the model for one epoch"""
    model.eval()
    
    # Initialize metric tracker
    metric_tracker = MetricTracker()
    loss_meter = AverageMeter()
    
    # Disable gradient computation for validation
    with torch.no_grad():
        progress_bar = tqdm(
            val_loader, desc=f"Validation Epoch {epoch+1}/{args.epochs}", 
            disable=rank != 0
        )
        
        for batch_idx, (rgb, depth_gt) in enumerate(progress_bar):
            # Move data to device
            rgb = rgb.to(device)
            depth_gt = depth_gt.to(device)
            
            # Ensure depth_gt is the right shape (B, H, W)
            if depth_gt.dim() == 4:
                depth_gt = depth_gt.squeeze(1)
            
            # Forward pass
            depth_pred = model(rgb)
            
            # Create mask for valid depth values
            mask = (depth_gt > 0) & torch.isfinite(depth_gt)
            
            # Compute loss
            loss = criterion(depth_pred, depth_gt, mask)
            loss_meter.update(loss.item())
            
            # Compute metrics - use prediction_ssi if available for aligned metrics
            if hasattr(criterion, 'prediction_ssi') and criterion.prediction_ssi is not None:
                aligned_pred = criterion.prediction_ssi
                batch_metrics = compute_metrics(aligned_pred, depth_gt, mask, align_depth=False)
            else:
                batch_metrics = compute_metrics(depth_pred, depth_gt, mask, align_depth=True)
            
            # Update metric tracker with batch metrics
            metric_tracker.update(batch_metrics, count=rgb.size(0))
            
            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}",
                    'abs_rel': f"{batch_metrics['abs_rel']:.4f}",
                    'delta1': f"{batch_metrics['delta1']:.4f}"
                })
            
            # Log images periodically
            if rank == 0 and batch_idx % args.log_freq == 0:
                # Visualize a few samples
                for i in range(min(2, rgb.size(0))):
                    # Choose the display prediction (aligned if available)
                    if hasattr(criterion, 'prediction_ssi') and criterion.prediction_ssi is not None:
                        display_pred = criterion.prediction_ssi[i]
                    else:
                        display_pred = depth_pred[i]
                    
                    # Log images with enhanced visualization
                    log_images_to_wandb(
                        rgb=rgb[i],
                        depth_gt=depth_gt[i],
                        depth_pred=display_pred,
                        phase='val',
                        sample_idx=i,
                        step=epoch * len(val_loader) + batch_idx
                    )
    
    # Get final metrics
    metrics = metric_tracker.get_metrics()
    
    # Add loss to metrics
    metrics['loss'] = loss_meter.avg
    
    # Log metrics
    if rank == 0:
        metric_log = {f'val/{k}': v for k, v in metrics.items()}
        metric_log['val/epoch'] = epoch
        wandb.log(metric_log)
        
        # Log detailed metrics
        logger.info(f"Validation Epoch {epoch+1} - "
                   f"Loss: {metrics['loss']:.4f}, "
                   f"RMSE: {metrics['rmse']:.4f}, "
                   f"Abs Rel: {metrics['abs_rel']:.4f}, "
                   f"Sq Rel: {metrics['sq_rel']:.4f}, "
                   f"RMSE log: {metrics['rmse_log']:.4f}, "
                   f"Delta1: {metrics['delta1']:.4f}, "
                   f"Delta2: {metrics['delta2']:.4f}, "
                   f"Delta3: {metrics['delta3']:.4f}")
    
    return metrics


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, 
                   device, epoch, args, rank, world_size):
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
        disable=rank != 0
    )
    
    for batch_idx, (rgb, depth_gt) in enumerate(progress_bar):
        # Move data to device
        rgb = rgb.to(device)
        depth_gt = depth_gt.to(device)
        
        # Ensure depth_gt is the right shape (B, H, W)
        if depth_gt.dim() == 4:
            depth_gt = depth_gt.squeeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        depth_pred = model(rgb)
        
        # Create mask for valid depth values
        mask = (depth_gt > 0) & torch.isfinite(depth_gt)
        
        # Compute loss
        loss = criterion(depth_pred, depth_gt, mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate if using step-based scheduler
        if scheduler is not None and args.scheduler != 'epoch':
            scheduler.step()
        
        # Update meters
        loss_meter.update(loss.item())
        
        # Update progress bar
        if rank == 0:
            progress_bar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'lr': f"{get_learning_rate(optimizer):.6f}"
            })
        
        # Log to wandb
        if rank == 0 and batch_idx % args.log_freq == 0:
            step = epoch * len(train_loader) + batch_idx
            wandb.log({
                'train/loss': loss_meter.val,
                'train/lr': get_learning_rate(optimizer)
            }, step=step)
            
            # Log images periodically
            if batch_idx % (10 * args.log_freq) == 0:
                # Visualize a few samples
                for i in range(min(2, rgb.size(0))):
                    # Choose the display prediction (aligned if available)
                    if hasattr(criterion, 'prediction_ssi') and criterion.prediction_ssi is not None:
                        display_pred = criterion.prediction_ssi[i]
                    else:
                        display_pred = depth_pred[i]
                    
                    # Log images with enhanced visualization
                    log_images_to_wandb(
                        rgb=rgb[i],
                        depth_gt=depth_gt[i],
                        depth_pred=display_pred,
                        phase='train',
                        sample_idx=i,
                        step=step
                    )
    
    # Update epoch-based scheduler
    if scheduler is not None and args.scheduler == 'epoch':
        scheduler.step()
    
    # Log epoch metrics
    if rank == 0:
        wandb.log({
            'train/epoch_loss': loss_meter.avg,
            'train/epoch': epoch
        })
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss_meter.avg:.4f}, "
                   f"LR: {get_learning_rate(optimizer):.6f}")


def train(args):
    """Main training function"""
    # Set up distributed training if available
    try:
        rank, world_size = ddp_setup()
        is_distributed = True
    except (KeyError, ValueError, RuntimeError):
        rank, world_size = 0, 1
        is_distributed = False
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb on rank 0
    if rank == 0:
        wandb.init(
            project="Depth-Anything-Relative",
            name=f"relative-depth-{args.encoder}-{args.loss_type}-lr{args.learning_rate}",
            config=vars(args)
        )
    
    # Create model
    model = RelativeDepthAnything(
        encoder=args.encoder,
        features=Config.features,
        out_channels=Config.out_channels,
        use_bn=Config.use_bn,
        pretrained_weights=args.pretrained_weights
    )
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Create dataloaders
    train_loader, train_dataset_size = get_dataloader(
        args, rank=rank, world_size=world_size, split='Train'
    )
    
    val_loader, _ = get_dataloader(
        args, rank=rank, world_size=world_size, split='Test'
    )
    
    # Store steps per epoch for scheduler
    args.train_steps_per_epoch = len(train_loader)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    # Create loss function
    criterion = get_loss_function(args)
    if rank == 0:
        logger.info(f"Using loss function: {args.loss_type}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if os.path.exists(ckpt_path):
            start_epoch, best_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler)
            if rank == 0:
                logger.info(f"Resuming from epoch {start_epoch} with best loss {best_loss:.6f}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, args, rank, world_size
        )
        
        # Validate
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # Use our implemented validate_one_epoch with the criterion
            metrics = validate_one_epoch(
                model, val_loader, criterion, device, epoch, args, rank, world_size
            )
            
            # Save best model
            if rank == 0:
                val_loss = metrics.get('abs_rel', float('inf'))
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(
                        model.module if is_distributed else model,
                        optimizer, scheduler, epoch, best_loss,
                        args.checkpoint_dir, 'best_model.pth'
                    )
                    logger.info(f"Saved best model with loss {best_loss:.6f}")
        
        # Save checkpoint periodically
        if rank == 0 and (epoch + 1) % args.save_ckpt_freq == 0:
            save_checkpoint(
                model.module if is_distributed else model,
                optimizer, scheduler, epoch, best_loss,
                args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    # Clean up
    if is_distributed:
        dist.destroy_process_group()
    
    if rank == 0:
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)