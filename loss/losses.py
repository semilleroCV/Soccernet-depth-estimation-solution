import torch
import torch.nn as nn
import torch.functional as F

from loss.utils import *

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
    def __init__(self, alpha=3.0, scales=4, reduction='batch-based'):
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

    def get_prediction_ssi(self):
        return self.prediction_ssi


class TrimmedMAELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return trimmed_mae_loss(prediction, target, mask, trim=0.2)

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=4.0, scales=4, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()

        self.data_loss = TrimmedMAELoss(reduction=reduction)
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.alpha = alpha

        self.prediction_ssi = None

    def forward(self, prediction, target, mask):
        self.prediction_ssi = normalize_prediction_robust(prediction, mask)
        target = normalize_prediction_robust(target, mask)

        total = self.data_loss(self.prediction_ssi, target, mask)
        if self.alpha > 0:
            total += self.alpha * self.regularization_loss(
                self.prediction_ssi, target, mask
            )

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, scales=4, alpha=4.0, reduction='batch-based'):
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()

        loss_reg = self.alpha * self.regularization_loss(pred, target, valid_mask)

        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))

        return loss + loss_reg

class ScaleAndShiftInvariantDALoss(nn.Module):
    def __init__(self, grad_matching, **kargs):
        super(ScaleAndShiftInvariantDALoss, self).__init__()
        self.grad_matching = grad_matching
        self.scaled_prediction = None

    def forward(self, prediction, target, mask, min_depth=None, max_depth=None, **kwargs):
        
        #_, h_i, w_i = prediction.shape
        #_, h_t, w_t = target.shape
    
        #if h_i != h_t or w_i != w_t:
        #    prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)

        #prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze().bool()
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return input * 0.0
        
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        
        self.scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        scale_target = target
        
        sampling_mask = mask
        if self.grad_matching:
            N = torch.sum(sampling_mask)
            d_diff = self.scaled_prediction - scale_target
            d_diff = torch.mul(d_diff, sampling_mask)

            v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
            v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
            h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
            loss = gradient_loss / N
        else:
            loss = nn.functional.l1_loss(self.scaled_prediction[mask], scale_target[mask])
            
        return loss

    def get_prediction_ssi(self):
        return self.scaled_prediction

def loss_dict():

    loss_fns = {
        'ssimse': ScaleAndShiftInvariantLoss(),
        'ssimse_nogradmatch': ScaleAndShiftInvariantLoss(alpha=0),
        'ssimae': TrimmedProcrustesLoss(),
        'ssigm': ScaleAndShiftInvariantDALoss(grad_matching=True),
        'silog': SiLogLoss(),
    }

    return loss_fns

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
        : int(len(res) * (1.0 - trim))
    ]

    return trimmed.sum() / (2 * M.sum())

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def compute_hdn_loss(SSI_LOSS, depth_preds, depth_gt, mask_valid_list):

    num_contexts = mask_valid_list.shape[0]
    repeated_preds = depth_preds.unsqueeze(0).repeat(num_contexts, 1, 1, 1, 1)
    repeated_gt = depth_gt.unsqueeze(0).repeat(num_contexts, 1, 1, 1, 1)
    
    repeated_preds = repeated_preds.reshape(-1, *depth_preds.shape[-3:])
    repeated_gt = repeated_gt.reshape(-1, *depth_gt.shape[-3:])
    repeated_masks = mask_valid_list.reshape(-1, *mask_valid_list.shape[-3:])
    
    hdn_loss_level = SSI_LOSS(repeated_preds, repeated_gt, repeated_masks)
    hdn_loss_level_list = hdn_loss_level.reshape(mask_valid_list.shape)
    hdn_loss_level_list = hdn_loss_level_list.sum(dim=0)
    
    mask_valid_list_times = mask_valid_list.sum(dim=0)
    
    valid_locations = (mask_valid_list_times != 0)
    hdn_loss_level_list[valid_locations] = (
        hdn_loss_level_list[valid_locations] / mask_valid_list_times[valid_locations]
    )
    
    final_loss = hdn_loss_level_list.sum() / mask_valid_list.sum()
    
    return final_loss