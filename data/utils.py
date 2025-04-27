import os
import glob
import numpy as np
import torch 

def find_rgb_depth(base_dir, split='Test'):
    """
    Find all RGB and corresponding depth images in specified split folders.
    Returns a list of tuples (rgb_image_path, depth_image_path).
    """
    image_pairs = []
    categories = ["depth-football", "depth-basketball"]

    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            continue

        split_path = os.path.join(category_path, split)
        if not os.path.exists(split_path):
            continue

        for game in os.listdir(split_path):
            game_path = os.path.join(split_path, game)
            if not os.path.isdir(game_path):
                continue

            for video_dir in os.listdir(game_path):
                if video_dir.startswith('video_'):
                    video_path = os.path.join(game_path, video_dir)
                    color_dir = os.path.join(video_path, "color")
                    depth_dir = os.path.join(video_path, "depth_r")

                    if os.path.exists(color_dir) and os.path.isdir(color_dir) and os.path.exists(depth_dir) and os.path.isdir(depth_dir):
                        # Find all PNG images in the color and depth directories
                        rgb_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
                        #depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

                        # Match RGB and depth images based on filename 
                        for rgb_img_path in rgb_files:
                            rgb_filename = os.path.basename(rgb_img_path)
                            depth_img_path = os.path.join(depth_dir, rgb_filename)

                            if os.path.exists(depth_img_path):
                                image_pairs.append((rgb_img_path, depth_img_path))
    return image_pairs

def normalize_image(image):

    min_val = np.min(image)
    max_val = np.max(image)

    normalized_img = (image - min_val) / (max_val - min_val)

    return normalized_img


def random_crop_pair(rgb_img, depth_img, mask, crop_size=518):

    H_rgb, W_rgb = rgb_img.shape[:2]
    H_depth, W_depth = depth_img.shape[:2]
    H_mask, W_mask = mask.shape[:2]
    
    if (H_rgb, W_rgb) != (H_depth, W_depth) or (H_rgb, W_rgb) != (H_mask, W_mask):
        raise ValueError("RGB, depth, and mask images must have the same spatial dimensions.")

    if crop_size > H_rgb or crop_size > W_rgb:
        raise ValueError(f"crop_size ({crop_size}) is larger than the image dimensions ({H_rgb}x{W_rgb}).")

    top = np.random.randint(0, H_rgb - crop_size + 1)
    left = np.random.randint(0, W_rgb - crop_size + 1)

    # Crop RGB image
    if rgb_img.ndim == 3:
        cropped_rgb = rgb_img[top:top + crop_size, left:left + crop_size, :]
    else:
        cropped_rgb = rgb_img[top:top + crop_size, left:left + crop_size]
    
    # Crop depth image
    if depth_img.ndim == 3:
        cropped_depth = depth_img[top:top + crop_size, left:left + crop_size, :]
    else:
        cropped_depth = depth_img[top:top + crop_size, left:left + crop_size]

    # Crop mask
    cropped_mask = mask[top:top + crop_size, left:left + crop_size]

    return cropped_rgb, cropped_depth, cropped_mask



def football_pitch_color_augmentation(image, preserve_green=True):
    """
    Apply color augmentation to football pitch images while preserving realistic colors.
    
    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W] with values in range [0, 1]
        preserve_green (bool): If True, applies less variation to the green channel to maintain
                              realistic grass appearance
    
    Returns:
        torch.Tensor: Color augmented image
    """

    if preserve_green:
        colors = torch.tensor([
            torch.FloatTensor(1).uniform_(0.8, 1.2), 
            torch.FloatTensor(1).uniform_(0.95, 1.05),  # green channel (less variation)
            torch.FloatTensor(1).uniform_(0.8, 1.2)    
        ]).to(image.device)
    else:
        colors = torch.FloatTensor(3).uniform_(0.9, 1.1).to(image.device)
    
    colors = colors.view(-1, 1, 1)
    
    image_aug = image * colors
    
    return image_aug