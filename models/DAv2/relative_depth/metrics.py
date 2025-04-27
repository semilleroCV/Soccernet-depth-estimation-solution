import torch
import numpy as np

def compute_mse(pred, target, mask=None):
    """
    Compute Mean Squared Error (MSE) between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        float: MSE value
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        
    if pred.numel() == 0:
        return 0.0
        
    return torch.mean((pred - target) ** 2).item()

def compute_rmse(pred, target, mask=None):
    """
    Compute Root Mean Squared Error (RMSE) between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        float: RMSE value
    """
    return torch.sqrt(torch.tensor(compute_mse(pred, target, mask))).item()

def compute_abs_rel(pred, target, mask=None):
    """
    Compute Absolute Relative error between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        float: Absolute Relative error
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        
    if pred.numel() == 0:
        return 0.0
        
    return torch.mean(torch.abs(pred - target) / (target + 1e-10)).item()

def compute_sq_rel(pred, target, mask=None):
    """
    Compute Squared Relative error between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        float: Squared Relative error
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        
    if pred.numel() == 0:
        return 0.0
        
    return torch.mean(((pred - target) ** 2) / (target + 1e-10)).item()

def compute_log10(pred, target, mask=None):
    """
    Compute log10 error between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        float: log10 error
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        
    if pred.numel() == 0:
        return 0.0
        
    return torch.mean(torch.abs(torch.log10(pred + 1e-10) - torch.log10(target + 1e-10))).item()

def compute_rmse_log(pred, target, mask=None):
    """
    Compute RMSE log error between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        float: RMSE log error
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        
    if pred.numel() == 0:
        return 0.0
        
    return torch.sqrt(torch.mean((torch.log(pred + 1e-10) - torch.log(target + 1e-10)) ** 2)).item()

def compute_threshold(pred, target, threshold=1.25, mask=None):
    """
    Compute threshold accuracy between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        threshold (float): Threshold value
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        float: Threshold accuracy
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        
    if pred.numel() == 0:
        return 0.0
        
    thresh = torch.max(pred / (target + 1e-10), target / (pred + 1e-10))
    return (thresh < threshold).float().mean().item()

def compute_scale_and_shift(pred, target, mask=None):
    """
    Compute optimal scale and shift for alignment.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        
    Returns:
        tuple: (scale, shift) factors
    """
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
    
    return scale, shift

def compute_metrics(pred, target, mask=None, align_depth=True):
    """
    Compute all metrics between prediction and target.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        mask (torch.Tensor, optional): Mask for valid pixels
        align_depth (bool): Whether to align prediction with target using scale and shift
        
    Returns:
        dict: Dictionary of metrics
    """
    # Align prediction with target if required
    if align_depth:
        scale, shift = compute_scale_and_shift(pred, target, mask)
        pred_aligned = torch.exp(scale) * pred + shift
    else:
        pred_aligned = pred
    
    # Compute metrics
    metrics = {
        'mse': compute_mse(pred_aligned, target, mask),
        'rmse': compute_rmse(pred_aligned, target, mask),
        'abs_rel': compute_abs_rel(pred_aligned, target, mask),
        'sq_rel': compute_sq_rel(pred_aligned, target, mask),
        'log10': compute_log10(pred_aligned, target, mask),
        'rmse_log': compute_rmse_log(pred_aligned, target, mask),
        'delta1': compute_threshold(pred_aligned, target, 1.25, mask),
        'delta2': compute_threshold(pred_aligned, target, 1.25**2, mask),
        'delta3': compute_threshold(pred_aligned, target, 1.25**3, mask)
    }
    
    return metrics

class MetricTracker:
    """
    Track metrics over multiple batches or epochs.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict, count=1):
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = 0
                self.counts[k] = 0
            
            self.metrics[k] += v * count
            self.counts[k] += count
    
    def get_metrics(self):
        return {k: self.metrics[k] / self.counts[k] for k in self.metrics}