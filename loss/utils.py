import torch
import numpy as np

def compute_scale_and_shift(prediction, target, mask):
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
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def reduction_batch_based(image_loss, M):

    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def reduction_image_based(image_loss, M):

    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)

def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


# Hierarchical Loss Function taken from: 

def get_contexts_dr( level, depth_gt, mask_valid):
    batch_norm_context = []
    for mask_index in range(depth_gt.shape[0]): #process each img in the batch
        depth_map = depth_gt[mask_index]
        valid_map = mask_valid[mask_index]

        if depth_map[valid_map].numel() == 0: #if there is no valid pixel
            map_context_list = [valid_map for _ in range(2 ** (level) - 1)]
        else:
            valid_values = depth_map[valid_map]
            max_d = valid_values.max()
            min_d = valid_values.min()
            bin_size_list = [(1 / 2) ** (i) for i in range(level)]
            bin_size_list.reverse()
            map_context_list = []
            for bin_size in bin_size_list:
                for i in range(int(1 / bin_size)):
                    mask_new = (depth_map >= min_d + (max_d - min_d) * i * bin_size) & (
                            depth_map < min_d + (max_d - min_d) * (i + 1) * bin_size + 1e-30)
                    mask_new = mask_new & valid_map
                    map_context_list.append(mask_new)
                    
        map_context_list = torch.stack(map_context_list, dim=0)
        batch_norm_context.append(map_context_list)
    batch_norm_context = torch.stack(batch_norm_context, dim=0).swapdims(0, 1)

    return batch_norm_context

def get_contexts_dp( level, depth_gt, mask_valid):

    depth_gt_nan=depth_gt.clone()
    depth_gt_nan[~mask_valid] = np.nan
    depth_gt_nan=depth_gt_nan.view(depth_gt_nan.shape[0], depth_gt_nan.shape[1], -1)

    bin_size_list = [(1 / 2) ** (i) for i in range(level)]
    bin_size_list.reverse()

    batch_norm_context=[]
    for bin_size in bin_size_list:
        num_bins=int(1/bin_size)

        for bin_index in range(num_bins):

            min_bin=depth_gt_nan.nanquantile(bin_index*bin_size,dim=-1).unsqueeze(-1).unsqueeze(-1)
            max_bin=depth_gt_nan.nanquantile((bin_index+1) * bin_size, dim=-1).unsqueeze(-1).unsqueeze(-1)

            new_mask_valid=mask_valid
            new_mask_valid=new_mask_valid &  (depth_gt>=min_bin)
            new_mask_valid = new_mask_valid & (depth_gt < max_bin)
            batch_norm_context.append(new_mask_valid)
    batch_norm_context = torch.stack(batch_norm_context, dim=0)
    return batch_norm_context

def init_temp_masks_ds(level,image_size):
    size=image_size
    bin_size_list = [(1 / 2) ** (i) for i in range(level)]
    bin_size_list.reverse()

    map_level_list = []
    for bin_size in bin_size_list:  # e.g. 1/8
        for h in range(int(1 / bin_size)):
            for w in range(int(1 / bin_size)):
                mask_new=torch.zeros(1,1,size,size)
                mask_new[:,:, int(h * bin_size * size):int((h + 1) * bin_size * size),
                int(w * bin_size * size):int((w + 1) * bin_size * size)] = 1
                mask_new = mask_new> 0
                map_level_list.append(mask_new)
    batch_norm_context=torch.stack(map_level_list,dim=0)
    return batch_norm_context

def get_contexts_ds( level, mask_valid):
    templete_contexts=init_temp_masks_ds(level,mask_valid.shape[-1])

    batch_norm_context = mask_valid.unsqueeze(0)
    batch_norm_context = batch_norm_context.repeat(templete_contexts.shape[0], 1, 1, 1, 1)
    batch_norm_context = batch_norm_context & templete_contexts

    return batch_norm_context

