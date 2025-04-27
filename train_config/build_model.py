import torch

from loguru import logger
from models.DAv2.depth_anything_v2.dpt import DepthAnythingV2

def build_depth_anything_v2(args):

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(f'checkpoints/DepthAnythingV2/depth_anything_v2_{args.encoder}.pth', map_location='cuda'))
    logger.info(f"Successfully loading depth anything v2 {args.encoder}.")

    return model
