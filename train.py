import os 
import math
import wandb
import random
import argparse
import numpy as np
import matplotlib.cm as cm

import torch
import torch.distributed as dist

from tqdm import tqdm
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler

from loss.losses import loss_dict
from data.soccernet_mde import get_dataloader
from metrics.metrics_challenge import RunningAverageDict, compute_metrics, aggregate_metrics, compute_scale_and_shift

from train_config.config import Config
from train_config.train_utils import (
    seed_all, get_optimizer, get_scheduler, save_checkpoint,
    load_checkpoint, visualize_depth, log_images, create_comparison_image,
    build_depth_anything_v2 )


def parse_tuple(s):
    s = s.strip('()')
    values = [int(x.strip()) for x in s.split(',')]
    return tuple(values)

def parse_args():

    parser = argparse.ArgumentParser(description='Soccernet Monocular Depth Estimation Challenge')

    ### Dataset ###
    parser.add_argument('--data_dir', type=str, default=Config.data_dir, help='Path to the dataset directory')
    parser.add_argument('--crop_size', type=parse_tuple, default=(546,966), 
                    help='Size for random crop during training (width,height), e.g. (1920,1080)')
    parser.add_argument('--sport', type=str, default=Config.sport, choices=['foot', 'basket'], help='Sport to evaluate the predictions on')

    ### Model ###
    parser.add_argument('--encoder', type=str, default=Config.encoder, choices=['vits', 'vitb', 'vitl', 'vitg'], help='Encoder to use')
    #parser.add_argument('--pretrained_weights', type=str, default=Config.pretrained_weights, help='Path to the pretrained weights')
    parser.add_argument('--load_head', action='store_true', help='Load head weights from pretrained model')

    ### Model Training ###
    parser.add_argument('--loss_fn', type=str, default=Config.loss_fn, choices=['ssimse', 'ssimae', 'silog', 'ssigm'], help='Loss function to use')
    parser.add_argument('--precision', type=str, default=Config.precision, choices=['float32', 'bfloat16', 'float16'])

    ### Model Hyperparameters ###
    parser.add_argument('--seed', type=int, default=Config.seed, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--learning_rate', type=float, default=Config.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=Config.weight_decay)
    parser.add_argument('--optimizer', type=str, default=Config.optimizer, choices=['adamw', 'adam'])
    parser.add_argument('--scheduler', type=str, default=Config.scheduler, choices=['cosine_warmup', 'cyclic' 'multi_step', 'custom', 'constant', 'linear_decay'])

    ### Checkpoints, validation and logging###
    parser.add_argument('--checkpoint_dir', type=str, default=Config.checkpoint_dir)
    parser.add_argument('--save_ckpt_freq', type=int, default=Config.save_ckpt_freq, help='Frequency to save checkpoints')
    parser.add_argument('--log_freq', type=int, default=Config.log_freq, help='Frequency to log images')
    parser.add_argument('--val_freq', type=int, default=Config.val_freq, help='Validation stage frequency')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging', default=False)
    
    ## Mask
    parser.add_argument('--mask', action='store_true', help='Use mask for training', default=False)
    parser.add_argument('--validate', action='store_true', help='Validate the model', default=False)

    args = parser.parse_args()

    return args

def ddp_setup():

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def train(args):
    #torch.autograd.set_detect_anomaly(True)

    rank, world_size = ddp_setup()
    seed_all(args.seed)

    if args.wandb and rank == 0:
        wandb.init(project='Soccernet MDE - DepthAnythingV2',
                   name=f'MDE {args.encoder}_epochs_{args.epochs}_lr_{args.learning_rate}_loss_{args.loss_fn}',
                   config=args,
                   mode='online')

    if args.validate:
        train_dataloader, len_train = get_dataloader(args, rank=rank, world_size=world_size, split='Train')
        val_dataloader, _ = get_dataloader(args, rank=rank, world_size=world_size, split='Test')
    else:
        train_dataloader, len_train = get_dataloader(args, rank=rank, world_size=world_size, split='Complete')

    model = build_depth_anything_v2(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, len_train, freq_ep=1, num_epochs=args.epochs)

    criterions = loss_dict()

    if args.loss_fn == 'ssigm':
        #criterion_ssi = criterions['ssimse_nogradmatch']
        criterion_ssi = criterions['ssimse']
        criterion_ssi_grad = criterions['ssigm']
    else:
        criterion_ssi = criterions[args.loss_fn]

    scaler = GradScaler()

    metric_evaluator = RunningAverageDict()
    start_epoch = 0
    best_metric_average = float("inf") 
    best_loss = float("inf")

    if args.resume:
        ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if os.path.exists(ckpt_path):
            start_epoch, best_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler)
            if rank == 0:
                logger.info(f"Resuming from epoch {start_epoch} with best loss {best_loss:.6f}")
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters()if p.requires_grad)
        total_params_millions = total_params / 1e6
        logger.info(f"Model has {total_params_millions:.2f} million parameters.")
        logger.info(f"Using {args.encoder} encoder with {args.crop_size} crop size.")
        logger.info(f"Using loss function {args.loss_fn}")
        logger.info(f"Using {args.optimizer} optimizer to train the model")
        logger.info(f"Using {args.scheduler} learning rate scheduler")

    for epoch_idx in range(start_epoch, args.epochs):

        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        # validation(args, rank, model, val_dataloader, metric_evaluator) # just to test things
        model.train()
        progress_bar = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc=f"Epoch {epoch_idx + 1}/{args.epochs}",
                            disable=(rank != 0))

        epoch_loss = 0.0

        for batch_idx, batch_sample in progress_bar:

            rgb, depth_gt, input_mask = [x.cuda(non_blocking=True) for x in batch_sample]
            depth_gt = depth_gt.squeeze(1)

            optimizer.zero_grad()
            
            depth_pred = model(rgb)  
            if args.mask:
                mask = input_mask.squeeze(1) * (depth_gt > 0.00001).float()
            else:
                mask = (depth_gt > 0).float()

            if args.loss_fn == 'ssigm':
                loss =  0.3 * criterion_ssi_grad(depth_pred, depth_gt, mask) + criterion_ssi(depth_pred, depth_gt, mask)
            else:
                loss = criterion_ssi(depth_pred, depth_gt, mask)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)

            optimizer.step()

            epoch_loss += loss.item()

            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'train_loss': loss.item(),
                    'lr': current_lr})

                gradient_norm = (sum(p.grad.detach().norm(2).item() ** 2 
                    for p in model.parameters() if p.grad is not None)) ** 0.5
                
                if args.wandb:
                    wandb.log({'Model grad norm': gradient_norm,
                            'SSI Loss': loss.item(),
                            "Learning_rate": current_lr})

                if batch_idx % args.log_freq == 0 and args.wandb and rank==0:
                    scaled_preds = criterion_ssi.get_prediction_ssi()
                    depth_gt = 1 - depth_gt 
                    scaled_preds = 1 - scaled_preds
                    log_images(rgb[0], depth_gt[0], scaled_preds[0], phase='Training')

            scheduler.step()

        epoch_loss_tensor = torch.tensor(epoch_loss, device='cuda')
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)

        avg_epoch_loss = epoch_loss_tensor.item() / (len_train * world_size)

        if args.wandb and rank == 0:
            wandb.log({
                'Train / Epoch Loss': avg_epoch_loss,
                'Train / Epoch': epoch_idx
            })
            
        if (epoch_idx + 1) % args.val_freq == 0 and args.validate:
            validation(args, rank, model, val_dataloader, metric_evaluator)
            if args.wandb and rank == 0:
                metrics = metric_evaluator.get_value()
                current_metric_average = aggregate_metrics(metrics)
                if current_metric_average < best_metric_average:
                    best_metric_average = current_metric_average
                    logger.info(f"New best metrics average: {current_metric_average:.4f} (previous: {best_metric_average:.4f}) - Saving checkpoint...")
                    save_checkpoint(args.checkpoint_dir, model, optimizer, scheduler, epoch_idx, avg_epoch_loss)
        
        if not args.validate:
            if (epoch_idx + 1) % args.save_ckpt_freq == 0 and rank == 0:
                save_checkpoint(args.checkpoint_dir, model, optimizer, scheduler, epoch_idx, avg_epoch_loss)

    torch.distributed.barrier()
    cleanup()

@torch.no_grad()
def validation(args, rank, model, val_loader, metric_evaluator, use_mask=False):

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", disable=(rank != 0))
    model.eval()

    for batch_idx, batch_sample in progress_bar:

        rgb, depth_gt, input_mask = [x.cuda(non_blocking=True) for x in batch_sample]
        depth_gt = depth_gt.squeeze()
        
        input_mask = input_mask.squeeze().float()  # Asegurar que tenga forma [B, H, W]

        depth_preds = model.module.infer_image(rgb)
        #print(f"max depth: {depth_preds.max()}, min depth: {depth_preds.min()}")
        if use_mask:
            mask = (depth_gt > 0).float() * input_mask
        else:
            mask = (depth_gt > 0).float()
            
        scale, shift = compute_scale_and_shift(depth_preds, depth_gt, mask)
        scaled_preds = scale.view(-1, 1, 1) * depth_preds + shift.view(-1, 1, 1)
        depth_gt = 1 - depth_gt
        scaled_preds = 1 - scaled_preds
        metric_evaluator.update(compute_metrics(depth_gt, scaled_preds, False, args.sport))
        torch.cuda.empty_cache()

        if batch_idx % 30 == 0 and args.wandb and rank==0:
            log_images(rgb[0], depth_gt[0], scaled_preds[0], phase='Validation')

    val_metrics = metric_evaluator.get_value()
    progress_bar.set_postfix({metric: f"{value:.4f}" for metric, value in val_metrics.items()})

    if args.wandb and rank==0:
        wandb.log(val_metrics)
    print("Evaluation completed.")
    print("\n".join(f"{k}: {v}" for k, v in val_metrics.items()))


def main():

    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args=args)

if __name__ == '__main__':
    main()