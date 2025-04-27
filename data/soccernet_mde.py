import random
import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms import functional as F
from data.utils import find_rgb_depth, normalize_image, random_crop_pair, football_pitch_color_augmentation

from loguru import logger
from PIL import Image

class SoccerNet(Dataset):
    def __init__(self, data_dir_path, 
                 crop_size=(546,966),
                 sport ='foot', 
                 split='Train'
                 ):
        super().__init__()

        self.split = split
        self.crop_size = crop_size
        self.sport = sport

        if split != 'Complete':
            self.image_pairs = find_rgb_depth(data_dir_path, split) #+ find_rgb_depth(data_dir_path, split='Validation') 
        else:
            self.image_pairs = find_rgb_depth(data_dir_path, split='Train') + find_rgb_depth(data_dir_path, split='Validation') + find_rgb_depth(data_dir_path, split='Test') + find_rgb_depth(data_dir_path, split='Extra')

        self.resize = transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BILINEAR) #1092,1932
        self.resize_target = transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.NEAREST_EXACT) #1092,1932
        self.to_tensor = transforms.Compose([transforms.ToImage(),
                                             transforms.ToDtype(torch.float32, scale=False)])

        self.rgb_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])                            

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):

        rgb_path, depth_path = self.image_pairs[idx]

        rgb_img = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32) / 255.0
        depth_img = np.array(Image.open(depth_path)).astype(np.float32) / 65535.0

        #depth_img = 1. / (depth_img + 1e-6)
        #depth_img = normalize_image(depth_img)

        depth_img = 1 - depth_img

        mask_path = rgb_path.replace("color", "masks").replace(".png", ".pt")
        mask = []
        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
        else:
            mask = torch.ones(*depth_img.shape, dtype=torch.uint8)
            
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (1, H, W)
        else:
            raise TypeError("mask must be a tensor")

        if self.split != 'Test':

            rgb_tensor = self.to_tensor(rgb_img)
            depth_tensor = self.to_tensor(depth_img)
            mask_tensor = self.to_tensor(mask)
            
            mask_tensor = self.resize_target(mask_tensor)       
            rgb_tensor = self.resize(rgb_tensor)
            depth_tensor = self.resize_target(depth_tensor)

            #cropped_rgb,cropped_depth, cropped_mask = random_crop_pair(rgb_img[:,:,:3], depth_img, mask, crop_size=self.crop_size)

        else:
            rgb_tensor = self.to_tensor(rgb_img)
            depth_tensor = self.to_tensor(depth_img)
            mask_tensor = self.to_tensor(mask)
            

        if self.split == 'Train' or self.split == 'Complete':

            do_hflip = np.random.uniform(0.0, 1.0)
            #do_vflip = np.random.uniform(0.0, 1.0)

            do_color_aug = np.random.uniform(0.0, 1.0)

            if do_color_aug > 0.5:

                # gamma augmentation
                gamma = np.random.uniform(0.8, 1.2)
                rgb_tensor = F.adjust_gamma(rgb_tensor, gamma)

                # brightness augmentation
                brightness = np.random.uniform(0.8, 1.2)
                rgb_tensor = F.adjust_brightness(rgb_tensor, brightness)
                
                # stauration augmentation
                saturation_factor = np.random.uniform(0.9, 1.1)
                rgb_tensor = F.adjust_saturation(rgb_tensor, saturation_factor)

                 # contrast augmentation
                contrast_factor = np.random.uniform(0.9, 1.1)
                rgb_tensor = F.adjust_contrast(rgb_tensor, contrast_factor)
                
                rgb_tensor = football_pitch_color_augmentation(rgb_tensor, preserve_green=True)

                rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)

            if do_hflip > 0.5:
                rgb_tensor = transforms.functional.hflip(rgb_tensor)
                depth_tensor = transforms.functional.hflip(depth_tensor)
                mask_tensor = transforms.functional.hflip(mask_tensor)

            #if do_vflip > 0.5:
            #    rgb_tensor = transforms.functional.vflip(rgb_tensor)
            #    depth_tensor = transforms.functional.vflip(depth_tensor)
            
        if self.split != 'Test':
            rgb_tensor = self.rgb_transform(rgb_tensor)
            
        return rgb_tensor, depth_tensor, mask_tensor


def get_dataloader(args, rank, world_size, split='Train'):

    #if split not in ['Train', 'Validation']:
    #    args.batch_size = 1
    #    args.crop_size = 1078

    if split=="Test":
        args.batch_size = 4

    shuffle = True if (split == 'Train' or split == 'Complete') else False

    dataset = SoccerNet(
        data_dir_path=args.data_dir,
        crop_size=args.crop_size,
        split=split
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        sampler=sampler,
        drop_last=False,
        pin_memory=True
    )

    if rank == 0:
        logger.info(f"Loading {len(dataset)} images from Soccernet {split} MDE dataset.")

    return loader, len(dataset)
