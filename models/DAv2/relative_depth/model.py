import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2

class RelativeDepthAnything(nn.Module):
    """
    Adaptation of Depth Anything V2 for relative depth estimation (0-1 range).
    """
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False,
        pretrained_weights=None
    ):
        super().__init__()
        
        # Initialize the base Depth Anything model
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Override configs if provided
        config = self.model_configs[encoder].copy()
        if features is not None:
            config['features'] = features
        if out_channels is not None:
            config['out_channels'] = out_channels
            
        # Initialize the model with a custom max_depth of 1.0 for relative depth
        self.depth_model = DepthAnythingV2(**config)
        
        # Replace the sigmoid in the output layer with a custom activation
        # that ensures output is in the 0-1 range
        self.depth_model.depth_head.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(config['features'] // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)  # Using ReLU to ensure positivity
        )
        
        # Load pretrained weights if provided
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
            
    def _load_pretrained(self, weights_path):
        """Load pretrained weights with custom handling for the modified output layer."""
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Filter out the output layer weights that we've modified
        filtered_state = {k: v for k, v in state_dict.items() 
                          if 'depth_head.scratch.output_conv2' not in k}
        
        # Load the filtered weights
        missing, unexpected = self.depth_model.load_state_dict(filtered_state, strict=False)
        
        # Log the loading results
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
            
    def forward(self, x):
        """Forward pass returning normalized depth in range 0-1."""
        depth = self.depth_model(x)
        
        # Normalize depth to 0-1 range if needed
        # depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth