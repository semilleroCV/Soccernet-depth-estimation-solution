import torch
import torch.nn as nn

# Implement missing utility functions

def reduction_batch_based(loss, M):
    """
    Batch-based reduction for loss values.
    
    Args:
        loss: Per-image loss values
        M: Number of valid pixels per image
    
    Returns:
        Reduced loss value
    """
    # Avoid division by zero
    valid = M > 0
    if valid.sum() > 0:
        return loss[valid].sum() / M[valid].sum()
    return torch.tensor(0.0, device=loss.device)

def reduction_image_based(loss, M):
    """
    Image-based reduction for loss values.
    
    Args:
        loss: Per-image loss values
        M: Number of valid pixels per image
    
    Returns:
        Reduced loss value
    """
    # Avoid division by zero
    valid = M > 0
    if valid.sum() > 0:
        return (loss[valid] / M[valid]).mean()
    return torch.tensor(0.0, device=loss.device)

def compute_scale_and_shift(prediction, target, mask):
    """
    Compute optimal scale and shift to align prediction with target.
    
    Args:
        prediction: Predicted depth
        target: Target depth
        mask: Mask for valid pixels
    
    Returns:
        tuple: (scale, shift)
    """
    # Apply mask if provided
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
    
    # Convert to log space
    prediction_log = torch.log(prediction + 1e-8)
    target_log = torch.log(target + 1e-8)
    
    # Compute means
    prediction_log_mean = prediction_log.mean()
    target_log_mean = target_log.mean()
    
    # Center log values
    prediction_log_centered = prediction_log - prediction_log_mean
    target_log_centered = target_log - target_log_mean
    
    # Compute scale and shift
    scale = (target_log_centered * prediction_log_centered).sum() / (prediction_log_centered ** 2).sum() + 1e-8
    shift = target_log_mean - scale * prediction_log_mean
    
    return torch.exp(scale), shift

def normalize_prediction_robust(prediction, mask):
    """
    Normalize prediction robustly.
    
    Args:
        prediction: Predicted depth
        mask: Mask for valid pixels
    
    Returns:
        Normalized prediction
    """
    # Apply mask if provided
    if mask is not None:
        valid_pred = prediction[mask]
    else:
        valid_pred = prediction
    
    # Skip normalization if no valid pixels
    if valid_pred.numel() == 0:
        return prediction
    
    # Compute robust min/max (trim outliers)
    n = valid_pred.numel()
    sorted_pred, _ = torch.sort(valid_pred.reshape(-1))
    min_val = sorted_pred[int(0.01 * n)]
    max_val = sorted_pred[int(0.99 * n)]
    
    # Normalize to [0, 1] range
    normalized = (prediction - min_val) / (max_val - min_val + 1e-8)
    
    # Clip values
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    return normalized


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

        self.scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.reduction)

        return total

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.data_loss = MSELoss(reduction=reduction)
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.alpha = alpha

        self.prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)

        self.prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.data_loss(self.prediction_ssi, target, mask)
        if self.alpha > 0:
            total += self.alpha * self.regularization_loss(self.prediction_ssi, target, mask)

        return total

class TrimmedMAELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return trimmed_mae_loss(prediction, target, mask, trim=0.2, reduction=self.reduction)

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()

        self.data_loss = TrimmedMAELoss(reduction=reduction)
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.alpha = alpha

        self.prediction_ssi = None

    def forward(self, prediction, target, mask):
        self.prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_normalized = normalize_prediction_robust(target, mask)

        total = self.data_loss(self.prediction_ssi, target_normalized, mask)
        if self.alpha > 0:
            total += self.alpha * self.regularization_loss(
                self.prediction_ssi, target_normalized, mask
            )

        return total


def loss_dict():
    loss_fns = {
        'ssimse': ScaleAndShiftInvariantLoss(),
        'ssimae': TrimmedProcrustesLoss()
    }

    return loss_fns

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def trimmed_mae_loss(prediction, target, mask, trim=0.2, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    # Get residuals only for masked values
    res = res[mask.bool()].abs()

    if res.numel() == 0:
        return torch.tensor(0.0, device=prediction.device)

    # Sort and trim residuals
    sorted_res, _ = torch.sort(res.reshape(-1))
    trimmed = sorted_res[:int(len(sorted_res) * (1.0 - trim))]

    # Calculate batch loss
    batch_loss = trimmed.sum()
    
    # Return according to reduction
    if reduction == reduction_batch_based:
        return batch_loss / (2 * M.sum())
    else:
        # For image-based, we need to distribute loss back to per-image
        # This is an approximation for trimmed loss
        return batch_loss / max(1, len(M))

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    # X gradient
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Y gradient
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def compute_hdn_loss(SSI_LOSS, depth_preds, depth_gt, mask_valid_list):
    """
    Compute Hierarchical Depth Networks loss.
    
    Args:
        SSI_LOSS: Scale and shift invariant loss function
        depth_preds: Predicted depth maps
        depth_gt: Ground truth depth maps
        mask_valid_list: List of masks for valid pixels
    
    Returns:
        Final loss value
    """
    num_contexts = mask_valid_list.shape[0]
    repeated_preds = depth_preds.unsqueeze(0).repeat(num_contexts, 1, 1, 1, 1)
    repeated_gt = depth_gt.unsqueeze(0).repeat(num_contexts, 1, 1, 1, 1)
    
    repeated_preds = repeated_preds.reshape(-1, *depth_preds.shape[-3:])
    repeated_gt = repeated_gt.reshape(-1, *depth_gt.shape[-3:])
    repeated_masks = mask_valid_list.reshape(-1, *mask_valid_list.shape[-3:])
    
    hdn_loss_level = SSI_LOSS(repeated_preds, repeated_gt, repeated_masks)
    
    # If the loss is a tensor, we need to reshape it
    if isinstance(hdn_loss_level, torch.Tensor) and hdn_loss_level.dim() == 1:
        hdn_loss_level_list = hdn_loss_level.reshape(mask_valid_list.shape[0], -1)
        hdn_loss_level_list = hdn_loss_level_list.sum(dim=0)
        
        mask_valid_list_times = mask_valid_list.sum(dim=0).bool().sum(dim=1).sum(dim=1)
        
        valid_locations = (mask_valid_list_times != 0)
        hdn_loss_level_list[valid_locations] = (
            hdn_loss_level_list[valid_locations] / mask_valid_list_times[valid_locations]
        )
        
        final_loss = hdn_loss_level_list.sum() / valid_locations.sum()
    else:
        # If the loss is already a scalar, just return it
        final_loss = hdn_loss_level
    
    return final_loss