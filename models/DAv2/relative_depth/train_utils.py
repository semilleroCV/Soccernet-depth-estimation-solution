import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import wandb

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_path, filename="checkpoint.pth"):
    """Save model checkpoint."""
    os.makedirs(save_path, exist_ok=True)
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss
    }
    torch.save(state, os.path.join(save_path, filename))

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')  # Return default values
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))

def get_learning_rate(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(optimizer, epoch, config):
    """Adjust learning rate according to schedule."""
    lr = config.learning_rate
    if config.lr_schedule == 'cosine':
        eta_min = lr * (config.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / config.epochs)) / 2
    else:  # step decay
        steps = np.sum(epoch > np.asarray(config.lr_decay_epochs))
        if steps > 0:
            lr = lr * (config.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def visualize_depth(depth, cmap='plasma'):
    """Visualize depth map with colormap."""
    depth_np = depth.detach().cpu().numpy()
    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    colormap = cm.get_cmap(cmap)
    depth_colored = colormap(depth_np)
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    return depth_colored

def log_metrics(metrics, step, prefix='train'):
    """Log metrics to wandb."""
    logged_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb.log(logged_metrics, step=step)

def log_images(rgb, depth_gt, depth_pred, step, max_samples=4):
    """Log images to wandb."""
    # Ensure we don't exceed batch size
    batch_size = min(rgb.shape[0], max_samples)
    
    for i in range(batch_size):
        # Convert tensors to numpy arrays
        rgb_np = rgb[i].permute(1, 2, 0).detach().cpu().numpy()
        depth_gt_np = depth_gt[i].detach().cpu().numpy()
        depth_pred_np = depth_pred[i].detach().cpu().numpy()
        
        # Normalize depth maps
        depth_gt_np = (depth_gt_np - depth_gt_np.min()) / (depth_gt_np.max() - depth_gt_np.min() + 1e-8)
        depth_pred_np = (depth_pred_np - depth_pred_np.min()) / (depth_pred_np.max() - depth_pred_np.min() + 1e-8)
        
        # Apply colormap
        colormap = cm.get_cmap('plasma')
        depth_gt_colored = colormap(depth_gt_np)
        depth_pred_colored = colormap(depth_pred_np)
        
        # Log images to wandb
        wandb.log({
            f"sample_{i}/rgb": wandb.Image(rgb_np),
            f"sample_{i}/depth_gt": wandb.Image(depth_gt_colored[:, :, :3]),
            f"sample_{i}/depth_pred": wandb.Image(depth_pred_colored[:, :, :3])
        }, step=step)

def compute_metrics(pred, target, mask=None):
    """Compute depth estimation metrics."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    # Scale and shift invariant alignment
    scale, shift = compute_scale_and_shift(pred, target)
    pred_aligned = scale * pred + shift
    
    # Compute metrics
    abs_rel = torch.mean(torch.abs(pred_aligned - target) / target)
    sq_rel = torch.mean(((pred_aligned - target) ** 2) / target)
    rmse = torch.sqrt(torch.mean((pred_aligned - target) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(pred_aligned) - torch.log(target)) ** 2))
    
    # Compute threshold metrics
    thresh = torch.max(target / pred_aligned, pred_aligned / target)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    
    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item()
    }

def compute_scale_and_shift(pred, target, mask=None):
    """Compute optimal scale and shift for alignment."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    # Convert to log space
    pred_log = torch.log(pred + 1e-8)
    target_log = torch.log(target + 1e-8)
    
    # Compute mean
    pred_log_mean = pred_log.mean()
    target_log_mean = target_log.mean()
    
    # Compute scale and shift
    pred_log_centered = pred_log - pred_log_mean
    target_log_centered = target_log - target_log_mean
    
    # Compute scale
    scale = (target_log_centered * pred_log_centered).sum() / (pred_log_centered ** 2).sum()
    
    # Compute shift
    shift = target_log_mean - scale * pred_log_mean
    
    # Convert back from log space
    scale = torch.exp(scale)
    shift = torch.exp(shift)
    
    return scale, shift

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricTracker:
    """Tracks multiple metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
    
    def update(self, metrics_dict):
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = AverageMeter()
            self.metrics[k].update(v)
    
    def get_metrics(self):
        return {k: meter.avg for k, meter in self.metrics.items()}