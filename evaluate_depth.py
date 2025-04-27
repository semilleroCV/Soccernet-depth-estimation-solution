import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import re
import json
import csv
from datetime import datetime
from metrics.metrics_challenge import compute_scale_and_shift, compute_metrics
from concurrent.futures import ThreadPoolExecutor
import gc

# Custom JSON encoder to handle NumPy and PyTorch types
class NumpyTorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NumpyTorchEncoder, self).default(obj)

def extract_frame_number(filename):
    """
    Extract the frame number from a filename safely.
    """
    base_name = os.path.splitext(filename)[0]
    if base_name.startswith('_'):
        base_name = base_name[1:]
    numbers = re.findall(r'\d+', base_name)
    return int(numbers[-1]) if numbers else None

def extract_path_info(pred_path, gt_path):
    """
    Extract game, video, and frame information from paths for submission format.
    """
    # Extract game and video information from the pred_path
    path_parts = pred_path.split(os.sep)
    
    # Find the index where "foot" appears
    for i, part in enumerate(path_parts):
        if "foot" in part.lower():
            category_idx = i
            break
    else:
        # If "foot" not found, try to extract info from filename
        category_idx = -1
    
    # In case the directory structure is different from expected
    # Try to extract the game and video info from filename pattern
    pred_filename = os.path.basename(pred_path)
    game_match = re.search(r'game_(\d+)', pred_filename)
    video_match = re.search(r'video_(\d+)', pred_filename)
    
    if game_match and video_match:
        game = game_match.group(1)
        video = video_match.group(1)
    else:
        try:
            # Assuming structure: .../foot/Test/game_X/video_Y/...
            if category_idx >= 0 and category_idx + 2 < len(path_parts) and category_idx + 3 < len(path_parts):
                game = path_parts[category_idx + 2].split('_')[-1]
                video = path_parts[category_idx + 3].split('_')[-1]
            else:
                # Fallback
                game = "1"
                video = "1"
        except (IndexError, ValueError):
            # Fallback
            game = "1"
            video = "1"
    
    # Extract frame number from ground truth path
    gt_filename = os.path.basename(gt_path)
    frame_num = extract_frame_number(gt_filename)
    
    return game, video, frame_num

def find_prediction_gt_pairs(pred_base_dir, gt_base_dir, split="Test"):
    """
    Find matching pairs of prediction and ground truth files.
    Returns a list of tuples (pred_path, frame_idx, gt_path, full_filename)
    """
    pairs = []
    
    # For football only
    category = "depth-football"
    pred_category_path = os.path.join(pred_base_dir, "foot", split)
    gt_category_path = os.path.join(gt_base_dir, split)
    
    print(f"Looking for prediction files in: {pred_category_path}")
    print(f"Looking for ground truth files in: {gt_category_path}")
    
    if not os.path.exists(pred_category_path):
        print(f"Warning: Prediction directory {pred_category_path} doesn't exist")
        pred_category_path = pred_base_dir  # Try using the base directory directly
        print(f"Trying with base directory: {pred_category_path}")

    if not os.path.exists(gt_category_path):
        print(f"Warning: Ground truth directory {gt_category_path} doesn't exist")
        return pairs
    
    # Check if directories exist and have content
    if not os.path.exists(pred_category_path):
        print(f"Error: Prediction directory {pred_category_path} doesn't exist")
        return pairs
        
    if not os.listdir(pred_category_path):
        print(f"Warning: No files found in {pred_category_path}")
        return pairs
    
    # Check prediction directory structure
    first_item = os.path.join(pred_category_path, os.listdir(pred_category_path)[0])
    if os.path.isdir(first_item):
        # Directory structure with game folders
        for game in os.listdir(pred_category_path):
            pred_game_path = os.path.join(pred_category_path, game)
            gt_game_path = os.path.join(gt_category_path, game)
            
            if not os.path.isdir(pred_game_path) or not os.path.isdir(gt_game_path):
                continue
                
            for video_dir in os.listdir(pred_game_path):
                pred_video_path = os.path.join(pred_game_path, video_dir)
                gt_video_path = os.path.join(gt_game_path, video_dir)
                
                if not os.path.isdir(pred_video_path) or not os.path.isdir(gt_video_path):
                    continue
                
                # Find ground truth depth files
                gt_depth_path = os.path.join(gt_video_path, "depth_r")
                if not os.path.exists(gt_depth_path):
                    print(f"Warning: No depth_r directory found for {gt_video_path}")
                    continue
                
                gt_files = []
                for gt_file in os.listdir(gt_depth_path):
                    if gt_file.endswith('.png') and not gt_file.startswith('._'):
                        frame_num = extract_frame_number(gt_file)
                        if frame_num is not None:
                            gt_files.append((frame_num, os.path.join(gt_depth_path, gt_file)))
                        else:
                            print(f"Warning: Could not extract frame number from {gt_file}")
                
                gt_files.sort(key=lambda x: x[0])
                
                # Match prediction files with ground truth
                for pred_file in os.listdir(pred_video_path):
                    if pred_file.endswith('.png'):
                        pred_path = os.path.join(pred_video_path, pred_file)
                        try:
                            frame_num = extract_frame_number(pred_file)
                            if frame_num is None:
                                print(f"Warning: Could not extract frame number from {pred_file}")
                                continue
                            
                            # Find matching ground truth
                            matching_gt = next(((num, gt_path) for num, gt_path in gt_files if num == frame_num), None)
                            if matching_gt is None:
                                print(f"Warning: No matching ground truth for frame {frame_num}")
                                continue
                            
                            _, gt_path = matching_gt
                            
                            # Create a filename for result tracking
                            full_filename = f"{category}_game_{game}_video_{video_dir}_color_{frame_num}.png"
                            
                            pairs.append((
                                pred_path,
                                frame_num,
                                gt_path,
                                full_filename
                            ))
                        except Exception as e:
                            print(f"Error processing prediction file {pred_path}: {e}")
    else:
        # Flat structure with all PNG files in one directory
        gt_files = []
        for game in os.listdir(gt_category_path):
            gt_game_path = os.path.join(gt_category_path, game)
            if not os.path.isdir(gt_game_path):
                continue
                
            for video_dir in os.listdir(gt_game_path):
                gt_video_path = os.path.join(gt_game_path, video_dir)
                if not os.path.isdir(gt_video_path):
                    continue
                    
                gt_depth_path = os.path.join(gt_video_path, "depth_r")
                if not os.path.exists(gt_depth_path):
                    continue
                    
                for gt_file in os.listdir(gt_depth_path):
                    if gt_file.endswith('.png') and not gt_file.startswith('._'):
                        frame_num = extract_frame_number(gt_file)
                        if frame_num is not None:
                            game_num = game.split('_')[-1] if '_' in game else game
                            video_num = video_dir.split('_')[-1] if '_' in video_dir else video_dir
                            gt_files.append((
                                frame_num,
                                os.path.join(gt_depth_path, gt_file),
                                game_num,
                                video_num
                            ))
        
        # Find matching prediction files
        for pred_file in os.listdir(pred_category_path):
            if pred_file.endswith('.png'):
                pred_path = os.path.join(pred_category_path, pred_file)
                
                try:
                    # Extract information from prediction filename
                    game_match = re.search(r'game_(\d+)', pred_file)
                    video_match = re.search(r'video_(\d+)', pred_file)
                    frame_match = re.search(r'depth_r_(\d+)', pred_file)
                    
                    if not (game_match and video_match and frame_match):
                        print(f"Warning: Could not extract info from {pred_file}")
                        continue
                        
                    game_num = game_match.group(1)
                    video_num = video_match.group(1)
                    frame_num = int(frame_match.group(1))
                    
                    # Find matching ground truth
                    matching_gt = next(((num, path, g, v) for num, path, g, v in gt_files 
                                      if num == frame_num and g == game_num and v == video_num), None)
                    
                    if matching_gt is None:
                        print(f"Warning: No matching ground truth for {pred_file}")
                        continue
                        
                    _, gt_path, _, _ = matching_gt
                    
                    full_filename = f"{category}_game_{game_num}_video_{video_num}_color_{frame_num}.png"
                    
                    pairs.append((
                        pred_path,
                        frame_num,
                        gt_path,
                        full_filename
                    ))
                    
                except Exception as e:
                    print(f"Error processing prediction file {pred_path}: {e}")
    
    print(f"Found {len(pairs)} matching prediction-ground truth pairs")
    return pairs

def load_prediction(pred_path, frame_idx):
    """
    Load depth prediction from a PNG file.
    
    Args:
        pred_path (str): Path to the PNG file.
        frame_idx (int): Unused for PNG predictions (one frame per file).
    
    Returns:
        torch.Tensor: A tensor with the prediction as float.
    """
    try:
        # Open the PNG file
        with Image.open(pred_path) as img:
            # Convert to numpy array and cast to float32
            depth_array = np.array(img).astype(np.float32)
            
        return torch.from_numpy(depth_array).float()
    except Exception as e:
        print(f"Error loading prediction {pred_path}: {e}")
        return None

def load_ground_truth(gt_path):
    """
    Load ground truth depth from PNG file.
    """
    try:
        img = Image.open(gt_path)
        # Convert to numpy array
        img_data = np.array(img)
        
        # Close the image file to free up file handles
        img.close()
        
        return torch.from_numpy(img_data).float()
    except Exception as e:
        print(f"Error loading ground truth {gt_path}: {e}")
        return None

def load_score_banner_files():
    """
    Load and parse the list of files that contain score banners.
    Returns a dictionary with keys (game, video, frame) for easier matching.
    """
    score_files_paths = []
    score_files_parsed = []
    
    try:
        with open('evaluation/test_score.txt', 'r') as f:
            score_files_paths = f.read().splitlines()
    except FileNotFoundError:
        try:
            with open('test_score.txt', 'r') as f:
                score_files_paths = f.read().splitlines()
        except FileNotFoundError:
            print("Warning: test_score.txt not found. Score banner masking will be disabled.")
            return score_files_parsed
    
    # Parse each file path to extract game, video, and frame numbers
    for file_path in score_files_paths:
        # Extract using regex
        game_match = re.search(r'game_(\d+)', file_path)
        video_match = re.search(r'video_(\d+)', file_path)
        frame_match = re.search(r'depth_r_(\d+)', file_path)
        
        if game_match and video_match and frame_match:
            game = game_match.group(1)
            video = video_match.group(1)
            frame = frame_match.group(1)
            
            # Store as a tuple for easy comparison
            score_files_parsed.append((game, video, frame))
    
    print(f"Parsed {len(score_files_parsed)} score banner files")
    return score_files_parsed

def save_submission_image(pred, pred_path, gt_path, submission_dir):
    """
    Save prediction as a 16-bit PNG for submission according to requirements.
    """
    try:
        # Extract necessary info for filename
        game, video, frame_num = extract_path_info(pred_path, gt_path)

        # Create filename according to competition format
        filename = f"foot_game_{game}_video_{video}_depth_r_{frame_num}.png"

        # Ensure submission directory exists
        os.makedirs(submission_dir, exist_ok=True)
        
        # Convert tensor to numpy
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().detach().numpy()
        else:
            pred_np = pred
            
        # Normalize to 0-1 range
        pred_min, pred_max = pred_np.min(), pred_np.max()
        pred_normalized = (pred_np - pred_min) / (pred_max - pred_min)
        
        # Scale to 16-bit range
        pred_scaled = (pred_normalized * 65535).astype(np.uint16)
        
        # Save as 16-bit PNG
        output_path = os.path.join(submission_dir, filename)
        Image.fromarray(pred_scaled).save(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error saving submission image: {e}")
        return None

def process_pair(pair, score_banner_files, mask, device, submission_dir=None):
    """
    Process a single prediction-ground truth pair.
    """
    pred_path, frame_idx, gt_path, full_filename = pair

    pred = load_prediction(pred_path, frame_idx)
    gt = load_ground_truth(gt_path)
    
    if pred is None or gt is None:
        return None
    
    # Check dimensions and resize if needed
    if pred.shape != gt.shape:
        try:
            # Handle different input dimensions
            if pred.dim() == 2:
                # If pred is already 2D (H,W), add batch and channel dims for interpolation
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(0).unsqueeze(0), 
                    size=(gt.shape[0], gt.shape[1]), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).squeeze(0)  # Remove the added dimensions
            else:
                # For other cases, try to adapt based on dimensions
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(0) if pred.dim() == 3 else pred, 
                    size=(gt.shape[0], gt.shape[1]), 
                    mode='bilinear', 
                    align_corners=False
                )
                # Ensure we get back to the right dimensions
                while pred.dim() > 2:
                    pred = pred.squeeze(0)
        except Exception as e:
            print(f"Error resizing prediction: {e}")
            print(f"pred shape: {pred.shape}, gt shape: {gt.shape}")
            return None
    
    pred = pred.to(device)
    gt = gt.to(device)
    
    gt = gt / 65535.0
    pred = pred / 65535.0
    
    if torch.all(gt == 1) or torch.all(gt == 0):
        return None
    
    # Make sure mask has the same dimensions as pred and gt
    if mask.shape != pred.shape:
        current_mask = torch.ones_like(pred, device=device)
    else:
        current_mask = mask.clone()
    
    # Extract game, video, frame from full_filename
    game_match = re.search(r'game_(\d+)', full_filename)
    video_match = re.search(r'video_(\d+)', full_filename)
    color_frame_match = re.search(r'color_(\d+)', full_filename)  # Using color_ instead of depth_r_

    game = game_match.group(1)
    video = video_match.group(1)
    frame = color_frame_match.group(1)
    
    # Check if the tuple (game, video, frame) exists in score_banner_files
    if (game, video, frame) in score_banner_files:
        # Adjust these coordinates based on the resized dimensions
        h, w = current_mask.shape
        bottom_region = int(70 / 1080 * h)
        top_region = int(122 / 1080 * h)
        left_region = int(95 / 1920 * w)
        right_region = int(612 / 1920 * w)
        #current_mask[bottom_region:top_region, left_region:right_region] = 0
        current_mask[70:122, 95:612] = 0

        mask_score = True

        print(f"Applying score banner mask for {game}, {video}, {frame}")
    else:
        mask_score = False
    
    try:
        # FIX: Add batch dimension to ensure 3D tensors for compute_scale_and_shift
        # The function expects tensors with shape [B, H, W]
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # Add batch dimension
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)  # Add batch dimension
        if current_mask.dim() == 2:
            current_mask = current_mask.unsqueeze(0)  # Add batch dimension
            
        # Compute scale and shift
        scale, shift = compute_scale_and_shift(pred, gt, current_mask)
        
        # FIX: Apply scale and shift safely
        # Make sure scale and shift are properly applied to the 2D pred tensor
        if torch.is_tensor(scale):
            # If scale is a tensor, ensure it's compatible with pred for broadcasting
            if scale.dim() == 0:  # Scalar tensor
                scaled_pred = scale.item() * pred + shift.item()
            else:
                # For tensors with dimensions, we need to reshape for proper broadcasting
                scaled_pred = scale * pred + shift
        else:
            # For scalar values
            scaled_pred = scale * pred + shift
        
        # FIX: Add batch dimension to both tensors before compute_metrics
        # The compute_metrics function expects 3D tensors with shape [B, H, W]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)  # Add batch dimension -> [1, H, W]
            
        if scaled_pred.dim() == 2:
            scaled_pred = scaled_pred.unsqueeze(0)  # Add batch dimension -> [1, H, W]
            
        # Compute metrics with properly formatted tensors
        metrics = compute_metrics(gt, scaled_pred, mask_score, "foot")
        
        # Save submission file if requested
        submission_path = None
        if submission_dir:
            submission_path = save_submission_image(scaled_pred, pred_path, gt_path, submission_dir)
        
        # Convert tensor metrics to native Python types before returning
        processed_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                processed_metrics[k] = float(v.item() if hasattr(v, 'item') else v)
            else:
                processed_metrics[k] = v
    except Exception as e:
        print(f"Error computing metrics for {pred_path}: {e}")
        print(f"Pred shape: {pred.shape}, GT shape: {gt.shape}")
        # Print more debug info
        if 'scale' in locals() and 'shift' in locals():
            print(f"Scale shape: {scale.shape if hasattr(scale, 'shape') else 'scalar'}, Shift shape: {shift.shape if hasattr(shift, 'shape') else 'scalar'}")
        if 'scaled_pred' in locals():
            print(f"Scaled pred shape: {scaled_pred.shape}")
        return None
    
    # Free memory
    del pred, gt, current_mask, scaled_pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'pred_path': pred_path,
        'gt_path': gt_path,
        'frame_idx': frame_idx,
        'metrics': processed_metrics,
        'submission_path': submission_path
    }

def save_metrics_csv(results, output_file):
    """
    Save metrics to a CSV file for submission and analysis.
    """
    fieldnames = ['filename', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            if result is None:
                continue
                
            metrics = result['metrics']
            submission_path = result.get('submission_path', '')
            if submission_path:
                filename = os.path.basename(submission_path)
            else:
                filename = 'unknown'
                
            writer.writerow({
                'filename': filename,
                'abs_rel': metrics.get('abs_rel', ''),
                'sq_rel': metrics.get('sq_rel', ''),
                'rmse': metrics.get('rmse', ''),
                'rmse_log': metrics.get('rmse_log', ''),
                'silog': metrics.get('silog', '')
            })

def evaluate_predictions(pred_base_dir, gt_base_dir, split="Test", batch_size=10, max_workers=1, generate_submission=False):
    """
    Evaluate depth predictions against ground truth and save results to a JSON file.
    Process in batches to manage memory usage.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    submission_dir = os.path.join("submission_files", datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) if generate_submission else None
    
    # Find matching prediction-ground truth pairs
    pairs = find_prediction_gt_pairs(pred_base_dir=pred_base_dir, gt_base_dir=gt_base_dir, split=split)
    
    if len(pairs) == 0:
        print("No prediction-ground truth pairs found. Check your directory structure.")
        return
    
    score_banner_files = load_score_banner_files()
    print(f"Found {len(score_banner_files)} files with score banners")
    
    mask = torch.ones(1080, 1920, device=device)
    
    football_metrics = {}
    all_metrics = {}
    
    detailed_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'split': split,
        'total_pairs': len(pairs),
        'device': str(device),
        'football': {'individual_results': [], 'average_metrics': {}},
        'overall': {'average_metrics': {}}
    }
    
    # Process in batches to avoid memory overload
    all_results = []
    
    # Start with single-threaded processing for easier debugging
    with tqdm(total=len(pairs), desc="Evaluating predictions") as pbar:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            
            # Process batch using multiple threads if max_workers > 1
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    batch_results = list(executor.map(
                        lambda pair: process_pair(pair, score_banner_files, mask, device, submission_dir), 
                        batch
                    ))
            else:
                # Single-threaded processing for debugging
                batch_results = []
                for pair in batch:
                    result = process_pair(pair, score_banner_files, mask, device, submission_dir)
                    batch_results.append(result)
            
            for result in batch_results:
                if result is not None:
                    all_results.append(result)
            
            # Update progress bar
            pbar.update(len(batch))
            
            # Periodically clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Process results
    for result in all_results:
        if result is None:
            continue
            
        pred_path = result['pred_path']
        gt_path = result['gt_path']
        frame_idx = result['frame_idx']
        metrics = result['metrics']
        
        result_entry = {
            'pred_path': pred_path,
            'gt_path': gt_path,
            'frame_idx': frame_idx,
            'metrics': metrics
        }
        
        detailed_results['football']['individual_results'].append(result_entry)
        for k, v in metrics.items():
            if k not in football_metrics:
                football_metrics[k] = []
            football_metrics[k].append(v)
        
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    # Calculate and store average metrics
    print("\nFootball Metrics:")
    if football_metrics:
        for k in football_metrics:
            avg_value = float(np.mean(football_metrics[k]))
            detailed_results['football']['average_metrics'][k] = avg_value
            print(f"{k}: {avg_value:.7f}")
    else:
        print("No football predictions evaluated")
    
    print("\nOverall Metrics:")
    if all_metrics:
        for k in all_metrics:
            avg_value = float(np.mean(all_metrics[k]))
            detailed_results['overall']['average_metrics'][k] = avg_value
            print(f"{k}: {avg_value:.7f}")
    else:
        print("No predictions evaluated")
    
    os.makedirs('evaluation_results', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join('evaluation_results', f'depth_evaluation_{split}_{timestamp}.json')
    
    try:
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, cls=NumpyTorchEncoder, indent=4)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"Error saving final results: {e}")
        # Try saving without indentation as a fallback
        try:
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, cls=NumpyTorchEncoder)
            print(f"Results saved without indentation to: {results_file}")
        except Exception as e2:
            print(f"Critical error: Could not save results at all: {e2}")
    
    # Save CSV metrics if requested
    if generate_submission:
        csv_file = os.path.join('evaluation_results', f'depth_metrics_{split}_{timestamp}.csv')
        save_metrics_csv(all_results, csv_file)
        print(f"Metrics saved to CSV: {csv_file}")
    
    return detailed_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate depth predictions against ground truth")
    parser.add_argument('--pred_dir', type=str, required=True, 
                        help='Path to prediction directory')
    parser.add_argument('--gt_dir', type=str, required=True, 
                        help='Path to ground truth directory')
    parser.add_argument('--split', type=str, default="Test", choices=["Test", "Validation"],
                        help='Dataset split to evaluate (Test or Validation)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of pairs to process in each batch')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Maximum number of worker threads')
    parser.add_argument('--generate_submission', action='store_true',
                        help='Generate submission files (PNG and CSV)')
    
    args = parser.parse_args()
    
    evaluate_predictions(
        pred_base_dir=args.pred_dir,
        gt_base_dir=args.gt_dir,
        split=args.split, 
        batch_size=args.batch_size, 
        max_workers=args.max_workers,
        generate_submission=args.generate_submission
    )

if __name__ == "__main__":
    main()
    # python evaluate_depth.py --pred_dir zoedepthpreds --gt_dir soccernet_data/depth-football/
    # python evaluate_depth.py --pred_dir preds --gt_dir soccernet_data/depth-football/