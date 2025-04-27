class Config:
    """Configuration for training relative depth estimation model."""
    
    # Model configuration
    encoder = 'vits'  # Options: 'vits', 'vitb', 'vitl', 'vitg'
    features = None  # If None, use default from encoder
    out_channels = None  # If None, use default from encoder
    use_bn = False
    pretrained_weights = '/ibex/user/perezpnf/SoccernetChallenge/checkpoints/DepthAnythingV2/depth_anything_v2_vitl.pth'  # Path to pretrained weights
    
    # Data configuration
    data_dir = 'soccernet_data'
    crop_size = 518
    sport = 'foot'
    
    # Training configuration
    batch_size = 8
    num_workers = 4
    epochs = 50
    precision = 'float16'  # Options: 'float32', 'float16', 'bfloat16'
    seed = 42
    
    # Optimizer configuration
    optimizer = 'adamw'  # Options: 'adam', 'adamw'
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Scheduler configuration
    scheduler = 'cosine_warmup'  # Options: 'cosine_warmup', 'cosine', 'step'
    warmup_epochs = 2
    min_lr = 1e-6
    lr_decay_rate = 0.1
    lr_decay_epochs = [20, 30]  # For step scheduler
    
    # Loss configuration
    loss_fn = 'ssimae' #Options 'ssimae', 'ssimse'
    scale_invariant_weight = 1.0
    ssim_weight = 1.0
    gradient_weight = 0.5
    
    # Checkpoint configuration
    checkpoint_dir = '/ibex/user/perezpnf/SoccernetChallenge/checkpoints'
    save_ckpt_freq = 5
    resume = False
    
    # Validation configuration
    val_freq = 1
    
    # Logging configuration
    log_freq = 100