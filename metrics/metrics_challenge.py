import torch
import numpy as np

def aggregate_metrics(metrics):
    total = (
        metrics["abs_rel"] +
        metrics["sq_rel"] +
        metrics["rmse"] +
        metrics["rmse_log"] +
        metrics["silog"]
    )
    return total / 5

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_scale_and_shift(prediction, target, mask):
    """
    Compute scale and shift to align the 'prediction' to the 'target' using the 'mask'.

    This function solves the system Ax = b to find the scale (x_0) and shift (x_1) that aligns the prediction to the target. 
    The system matrix A and the right hand side b are computed from the prediction, target, and mask.

    Args:
        prediction (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        mask (torch.Tensor): Mask that indicates the zones to evaluate. 

    Returns:
        tuple: Tuple containing the following:
            x_0 (torch.Tensor): Scale factor to align the prediction to the target.
            x_1 (torch.Tensor): Shift to align the prediction to the target.
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0
    
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def compute_errors(gt, pred):
    """
    Compute the 5 error metrics between the ground truth and the prediction:
    - Absolute relative error (abs_rel)
    - Squared relative error (sq_rel)
    - Root mean squared error (rmse)
    - Root mean squared error on the log scale (rmse_log)
    - Scale invariant log error (silog)

    Args:
        gt (numpy.ndarray): Ground truth values.
        pred (numpy.ndarray): Predicted values.

    Returns:
        dict: Dictionary containing the following metrics:
            'abs_rel': Absolute relative error
            'sq_rel': Squared relative error
            'rmse': Root mean squared error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """


    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    return dict(abs_rel=abs_rel, rmse=rmse, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_metrics(gt, pred, mask_score, sport):
    """
    Computes averaged error metrics for a batch of depth maps.

    Args:
        gt (torch.Tensor): Ground truth tensor of shape [B, H, W].
        pred (torch.Tensor): Prediction tensor of shape [B, H, W].
        mask_score (bool): Whether to mask a score area (for "foot" sport).
        sport (str): The sport to evaluate ("basket" or "foot").

    Returns:
        dict: Averaged error metrics computed via compute_errors on valid pixels.
    """
    # Convert tensors to numpy arrays.
    gt_np = gt.detach().cpu().numpy()  # shape: [B, H, W]
    pred_np = pred.detach().cpu().numpy()  # shape: [B, H, W]

    # invert them
    #gt_np = 1. / (gt_np + 1e-7)
    #pred_np = 1. / (gt_np + 1e-7)
    
    B, H, W = gt_np.shape
    errors_list = []
    
    for i in range(B):
        # Start with a full valid mask.
        mask = np.ones((H, W), dtype=bool)
        
        # Apply sport-specific mask modifications.
        if sport == "basket":
            mask[870:1016, 1570:1829] = False
        elif sport == "foot" and mask_score:
            mask[70:122, 95:612] = False
        
        # Get the i-th prediction and ground truth.
        g = gt_np[i]
        p = pred_np[i]
        
        # Invalidate pixels where prediction or ground truth is non-positive, inf, or NaN.
        mask[p <= 0] = False
        mask[np.isinf(p)] = False
        mask[np.isnan(p)] = False
        
        mask[g <= 0] = False
        mask[np.isinf(g)] = False
        mask[np.isnan(g)] = False
        
        # Compute errors for the current sample.
        errors = compute_errors(g[mask], p[mask])
        errors_list.append(errors)
    
    # Average each metric across the batch.
    avg_errors = {}
    for key in errors_list[0]:
        avg_errors[key] = np.mean([e[key] for e in errors_list])
    
    return avg_errors
