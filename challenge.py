import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
from loguru import logger

from train_config.config import Config
from train_config.train_utils import build_depth_anything_v2
from torchvision.transforms import v2 as transforms


import cv2

import traceback

rgb_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])    

def parse_args():
    parser = argparse.ArgumentParser(description='Soccernet MDE Challenge Inference')
    
    parser.add_argument('--encoder', type=str, default=Config.encoder, 
                        choices=['vits', 'vitb', 'vitl', 'vitg'], 
                        help='Encoder to use')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the challenge dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save prediction results')
    
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], 
                        help='Device to run inference on')
    parser.add_argument('--precision', type=str, default='float32', 
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--load_head', action='store_true', 
                        help='Load head weights', default=True)
    
    args = parser.parse_args()
    return args

def extract_frame_number(filename):
    """
    Extract the frame number from a filename.
    """
    base_name = os.path.splitext(filename)[0]
    if base_name.startswith('_'):
        base_name = base_name[1:]
    numbers = re.findall(r'\d+', base_name)
    return int(numbers[-1]) if numbers else None

def find_challenge_images(input_dir):
    """
    Find all RGB images in the challenge dataset with proper structure information.
    Returns a list of tuples (image_path, game_number, video_number, frame_number)
    """
    images = []
    
    base_dir = input_dir
    if os.path.exists(os.path.join(input_dir, 'foot')):
        base_dir = os.path.join(input_dir, 'foot')
    if os.path.exists(os.path.join(base_dir, 'challenge')):
        base_dir = os.path.join(base_dir, 'challenge')
    
    logger.info(f"Searching for challenge images in: {base_dir}")
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            img_path = os.path.join(root, file)
            path_parts = img_path.split(os.sep)

            if (file.endswith('.png') and 
                not file.startswith('._') and 
                'depth' not in file.lower() and 
                any(part.lower() == 'color' for part in path_parts)
                and any(part.lower() == 'video_1' for part in path_parts)):

                
                game_num = None
                video_num = None
                frame_num = None
                
                for part in path_parts:
                    if part.startswith('game_'):
                        game_num = part.split('_')[-1]
                    elif part.startswith('video_'):
                        video_num = part.split('_')[-1]
                
                frame_num = extract_frame_number(file)
                
                if not all([game_num, video_num, frame_num]):
                    filename = os.path.basename(img_path)
                    game_match = re.search(r'game_(\d+)', filename)
                    video_match = re.search(r'video_(\d+)', filename)
                    
                    if game_match and not game_num:
                        game_num = game_match.group(1)
                    if video_match and not video_num:
                        video_num = video_match.group(1)
                    
                    if not frame_num:
                        color_match = re.search(r'color_(\d+)', filename)
                        rgb_match = re.search(r'rgb_(\d+)', filename)
                        
                        if color_match:
                            frame_num = int(color_match.group(1))
                        elif rgb_match:
                            frame_num = int(rgb_match.group(1))
                
                if game_num and video_num and frame_num:
                    images.append((img_path, game_num, video_num, frame_num))
                else:
                    logger.warning(f"Could not extract info from {img_path}")
    
    if not images:
        raise ValueError(f"No suitable images found in {input_dir}. Check directory structure.")
    
    logger.info(f"Found {len(images)} images for inference")
    return images
    
@torch.no_grad()
def process_batch(img_paths, model, device, precision,
                  pad_input=True, fh=3, fw=3, 
                  upsampling_mode='bicubic', padding_mode="reflect"):
    """
    Process a batch of images and return depth predictions at original resolutions
    using padding augmentation to mitigate boundary artifacts.
    
    Args:
        img_paths (list[str]): List of file paths to the input images.
        model: The depth estimation model.
        device: Torch device to run inference on.
        precision (str): Inference precision (e.g. 'float16' or else).
        pad_input (bool, optional): Whether to pad the input for augmentation.
                                    Defaults to True.
        fh (float, optional): Vertical padding factor. Calculated as: int(sqrt(h/2) * fh).
                              Defaults to 3.
        fw (float, optional): Horizontal padding factor. Calculated as: int(sqrt(w/2) * fw).
                              Defaults to 3.
        upsampling_mode (str, optional): Interpolation mode if output size does not match.
                                         Defaults to 'bicubic'.
        padding_mode (str, optional): Padding mode to be used by torch.nn.functional.pad.
                                      Defaults to "reflect".
    
    Returns:
        list[torch.Tensor]: List of depth prediction tensors cropped to original image resolutions.
    """
    batch = []

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        if img is None:
            raise ValueError(f"Unable to load image from {img_path}")

        img_tensor = rgb_transform(img)

        batch.append(img_tensor)
        
    batch = torch.stack(batch).to(device).contiguous()
    orig_h, orig_w = batch.shape[-2], batch.shape[-1]

    depth_preds = model.infer_image(batch, w=orig_w, h=orig_h, interpolate_mode="bicubic", flip_aug=True, with_pad=False)

    return depth_preds

def save_depth_prediction(depth_pred, game_num, video_num, frame_num, output_dir):
    """
    Save depth prediction as a 16-bit PNG file with the required naming convention.
    """
    
    filename = f"foot_game_{game_num}_video_{video_num}_depth_r_{frame_num}.png"
    output_path = os.path.join(output_dir, filename)
    
    if isinstance(depth_pred, torch.Tensor): 
        depth_np = depth_pred.cpu().numpy()
    else:
        depth_np = depth_pred

    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

    depth_np = 1 - depth_np

    depth_recovered = depth_np  * 65535.0

    depth_recovered = depth_recovered.astype(np.uint16)
    
    Image.fromarray(depth_recovered).save(output_path)
    
    return output_path

def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = build_depth_anything_v2(args)
    
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
 
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    challenge_images = find_challenge_images(args.input_dir)
    batch_size = 6

    with torch.no_grad():
        for i in tqdm(range(0, len(challenge_images), batch_size), desc="Processing batches"):
            batch_images = challenge_images[i:i+batch_size]
            img_paths = [img[0] for img in batch_images]
            
            try:
                depth_preds = process_batch(img_paths, model, device, args.precision)

                for j, (_, game_num, video_num, frame_num) in enumerate(batch_images):
                    output_path = save_depth_prediction(depth_preds[j], game_num, video_num, frame_num, args.output_dir)
                    logger.debug(f"Saved prediction to {output_path}")
            except Exception as e:
                logger.error(f"Error processing batch starting with {img_paths[0]}: {e}")
                # Print more detailed error information
                logger.error(traceback.format_exc())
    
    logger.info(f"Inference completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()

    # python challenge.py --encoder vitl --checkpoint_path checkpoints/complete/checkpoint_epoch_05.ckpt --input_dir soccernet_data/depth-football/Challenge --output_dir preds2
    # python challenge.py --encoder vitb --checkpoint_path checkpoints/checkpoint_epoch_07.ckpt --input_dir soccernet_data/depth-football/Challenge --output_dir preds3
    # python challenge.py --encoder vitb --checkpoint_path checkpoints/checkpoint_epoch_03.ckpt --input_dir soccernet_data/depth-football/Test --output_dir preds/foot/Test