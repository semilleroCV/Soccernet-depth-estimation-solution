import os
import math
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from loguru import logger
from matplotlib import cm
from torch.optim.lr_scheduler import LambdaLR
from models.DAv2.depth_anything_v2.dpt import DepthAnythingV2
from transformers import get_cosine_schedule_with_warmup

def seed_all(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(ckpt_dir, model, optimizer, scheduler, epoch, best_loss, rank=0):
    if rank != 0:
        return None

    os.makedirs(ckpt_dir, exist_ok=True)
    
    # In DDP (or FSDP), the underlying model is stored in model.module
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'best_loss': best_loss
    }

    ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1:02d}.ckpt')
    torch.save(checkpoint, ckpt_path)
    
    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, ckpt_path, weights_only=False, device='cpu', load_head=False):
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    if not weights_only:
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler_state = checkpoint.get('scheduler')
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    last_epoch = checkpoint.get('epoch', 0)
    
    del checkpoint
    torch.cuda.empty_cache()
    
    return last_epoch


def get_optimizer(args, model):
    
    optimizers = {
        'adamw': optim.AdamW,
        'adam': optim.Adam
    }

    if args.optimizer not in optimizers:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Layer {name} is frozen and will not be updated during training.")

    optimizer_params = {
        'params': [{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.learning_rate },
                    {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.learning_rate * 5 }],
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
        'betas': (0.9, 0.999)
    }

    optimizer = optimizers[args.optimizer](**optimizer_params)

    return optimizer


def get_scheduler(args, optimizer, len_train, freq_ep=1,
                  num_epochs=10, warmup_fraction=0.1,
                  warmup_start_lr=2e-5, max_lr=1e-3, final_lr=2e-4):
    
    steps_per_epoch = math.ceil(len_train / args.batch_size)

    schedulers = {
        'cosine_warmup': get_cosine_schedule_with_warmup,
        'cyclic': optim.lr_scheduler.CyclicLR,
        'multi_step': optim.lr_scheduler.MultiStepLR,
        'custom_lambda': optim.lr_scheduler.LambdaLR,
        'constant': optim.lr_scheduler.LambdaLR,
        'linear_decay': optim.lr_scheduler.LambdaLR,           # ← añadimos
    }

    scheduler_params = {
        'cosine_warmup': {
            'num_warmup_steps': int(warmup_fraction * steps_per_epoch * num_epochs),
            'num_training_steps': int(steps_per_epoch * num_epochs)
        },
        'cyclic': {
            'base_lr': 1e-6,
            'max_lr': 1e-4,
            'step_size_up': steps_per_epoch * freq_ep,
            'cycle_momentum': False
        },
        'multi_step': {
            'milestones': [10, 20, 30, 40],
            'gamma': 0.1
        },
        'custom_lambda': {
            'warmup_iters': int(warmup_fraction * steps_per_epoch * num_epochs),
            'total_iters': steps_per_epoch * num_epochs,
            'warmup_start_lr': warmup_start_lr,
            'max_lr': max_lr,
            'final_lr': final_lr
        },
        'constant': {
            'lr_value': args.lr if hasattr(args, 'lr') else max_lr
        },
        'linear_decay': {}    # no params externos
    }
    
    if args.scheduler not in schedulers:
        raise ValueError(f"Invalid scheduler: {args.scheduler}")

    if args.scheduler == 'custom_lambda':
        p = scheduler_params['custom_lambda']
        lr_lambda = create_lr_lambda(
            p['warmup_iters'], p['total_iters'],
            p['warmup_start_lr'], p['max_lr'], p['final_lr']
        )
        lr_scheduler = schedulers['custom_lambda'](optimizer, lr_lambda=lr_lambda)

    elif args.scheduler == 'constant':
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0
        )

    elif args.scheduler == 'linear_decay':
        total_steps  = steps_per_epoch * num_epochs
        start_decay  = steps_per_epoch * 3

        def lr_lambda(step):
            if step < start_decay:
                return 1.0
            # progreso normalizado [0,1] entre start_decay y total_steps
            progress = (step - start_decay) / max(total_steps - start_decay, 1)
            progress = min(progress, 1.0)
            # lineal de 1.0 a 0.5
            return 1.0 - 0.5 * progress

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        params = scheduler_params[args.scheduler]
        lr_scheduler = schedulers[args.scheduler](optimizer, **params)

    return lr_scheduler



def create_lr_lambda(warmup_iters, total_iters, warmup_start_lr, max_lr, final_lr):
    def lr_lambda(iteration):
        if iteration < warmup_iters:
            # Linear warmup scaling factor
            return 1 + (max_lr / warmup_start_lr - 1) * (iteration / warmup_iters)
        else:
            # Cosine annealing scaling factor
            progress = (iteration - warmup_iters) / (total_iters - warmup_iters)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (final_lr + (max_lr - final_lr) * cosine_decay) / warmup_start_lr
    return lr_lambda


def visualize_depth(depth, cmap='Spectral_r'):

    depth = depth.detach().cpu().numpy().squeeze()  
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    colormap = cm.get_cmap(cmap)
    depth_color = colormap(depth)
    depth_color = (depth_color[:, :, :3] * 255).astype(np.uint8)

    return depth_color


def log_images(rgb, depth_gt, depth_pred, phase):
    """Log images to wandb."""

    rgb_img = rgb.permute(1, 2, 0).detach().cpu().numpy()
    
    # Create error map
    valid_mask = (depth_gt > 0) & torch.isfinite(depth_gt)
    error_map = torch.zeros_like(depth_gt)
    if valid_mask.any():
        # Compute absolute relative error
        abs_rel_error = torch.abs(depth_pred - depth_gt) / (depth_gt + 1e-10)
        error_map[valid_mask] = abs_rel_error[valid_mask].detach()
        #error_map = error_map.cpu().numpy()
    
    # Create comparison image
    depth_gt = visualize_depth(depth_gt)
    depth_pred = visualize_depth(depth_pred)
    comparison_img = create_comparison_image(rgb_img, depth_gt, depth_pred, error_map)

    wandb.log({
        f'{phase}/rgb': wandb.Image(rgb_img),
        f'{phase}/depth_gt': wandb.Image(depth_gt),
        f'{phase}/depth_pred': wandb.Image(depth_pred),
        f'{phase}/depth_error': wandb.Image(visualize_depth(error_map)),
        f'{phase}/comparison': wandb.Image(comparison_img)
    })

    plt.close('all')


def create_comparison_image(rgb, depth_gt, depth_pred, error_map=None, cmap='Spectral_r'):
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

    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    if error_map is not None:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(rgb) 
    axs[0].set_title('RGB Image')
    axs[0].axis('off')

    #vmin, vmax = np.nanpercentile(depth_gt[depth_gt > 0], (1, 99))
    im_gt = axs[1].imshow(depth_gt, cmap=cmap)
    axs[1].set_title('Ground Truth Depth')
    axs[1].axis('off')
    plt.colorbar(im_gt, ax=axs[1], fraction=0.046, pad=0.04)

    im_pred = axs[2].imshow(depth_pred, cmap=cmap)
    axs[2].set_title('Predicted Depth')
    axs[2].axis('off')
    plt.colorbar(im_pred, ax=axs[2], fraction=0.046, pad=0.04)

    if error_map is not None:
        #vmin_err, vmax_err = np.nanpercentile(error_np[depth_gt > 0], (1, 99))
        im_err = axs[3].imshow(error_map.detach().cpu().numpy(), cmap=cmap)
        axs[3].set_title('Error Map')
        axs[3].axis('off')
        plt.colorbar(im_err, ax=axs[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    return fig


def build_depth_anything_v2(args):

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[args.encoder])
    if args.load_head:
        model.load_state_dict(torch.load(f'checkpoints/DepthAnythingV2/depth_anything_v2_{args.encoder}.pth', map_location='cuda'))
    else:
        print("Loading pretrained weights for encoder only.")
        model.load_state_dict({k: v for k, v in torch.load(f'checkpoints/DepthAnythingV2/depth_anything_v2_{args.encoder}.pth', map_location='cuda').items() if 'pretrained' in k}, strict=False)

    logger.info(f"Successfully loading depth anything v2 {args.encoder}.")

    return model

