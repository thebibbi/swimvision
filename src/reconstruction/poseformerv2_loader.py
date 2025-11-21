"""
PoseFormerV2 Model Loader
=========================

Utility functions to load PoseFormerV2 models from the repository.
Uses frequency-domain processing (DCT) for robust 3D pose estimation.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Add PoseFormerV2 repo to path
POSEFORMERV2_PATH = Path(__file__).parent.parent.parent / "models" / "poseformerv2"
sys.path.insert(0, str(POSEFORMERV2_PATH))

# Import PoseFormerV2 model
try:
    from common.model_poseformer import PoseTransformerV2
    POSEFORMERV2_AVAILABLE = True
except ImportError:
    logger.warning(f"Failed to import PoseFormerV2 from {POSEFORMERV2_PATH}")
    POSEFORMERV2_AVAILABLE = False


# Model configurations for different variants
# Format: frames_kept-frames_total-mpjpe
MODEL_CONFIGS = {
    '1-27-48.7': {
        'num_frame': 27,
        'frame_kept': 1,
        'coeff_kept': 3,
        'depth': 4,
        'embed_dim': 32,
        'num_heads': 8,
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
    },
    '3-27-47.9': {
        'num_frame': 27,
        'frame_kept': 3,
        'coeff_kept': 3,
        'depth': 4,
        'embed_dim': 32,
        'num_heads': 8,
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
    },
    '1-81-47.6': {
        'num_frame': 81,
        'frame_kept': 1,
        'coeff_kept': 3,
        'depth': 4,
        'embed_dim': 32,
        'num_heads': 8,
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
    },
    '3-81-47.1': {
        'num_frame': 81,
        'frame_kept': 3,
        'coeff_kept': 3,
        'depth': 4,
        'embed_dim': 32,
        'num_heads': 8,
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
    },
    '9-81-46.0': {
        'num_frame': 81,
        'frame_kept': 9,
        'coeff_kept': 9,
        'depth': 4,
        'embed_dim': 32,
        'num_heads': 8,
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
    },
    '27-243-45.2': {  # Best model
        'num_frame': 243,
        'frame_kept': 27,
        'coeff_kept': 27,
        'depth': 4,
        'embed_dim': 32,
        'num_heads': 8,
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
    },
}

# Pre-trained weights download info
PRETRAINED_WEIGHTS = {
    '1-27-48.7': {
        'gdown_id': '14J0GYIzk_rGKSMxAPI2ydzX76QB70-g3',
        'filename': '1_27_48.7.bin',
    },
    '3-27-47.9': {
        'gdown_id': '13oJz5-aBVvvPVFvTU_PrLG_m6kdbQkYs',
        'filename': '3_27_47.9.bin',
    },
    '1-81-47.6': {
        'gdown_id': '14WgFFBsP0DtTq61XZWI9X2TzvFLCWEnd',
        'filename': '1_81_47.6.bin',
    },
    '3-81-47.1': {
        'gdown_id': '13rXCkYnVnkbT-cz4XCo0QkUnUEYiSeoi',
        'filename': '3_81_47.1.bin',
    },
    '9-81-46.0': {
        'gdown_id': '13wla4b5RgJGKX5zVehv4qKhCrQEFhfzG',
        'filename': '9_81_46.0.bin',
    },
    '27-243-45.2': {  # Best model
        'gdown_id': '14SpqPyq9yiblCzTH5CorymKCUsXapmkg',
        'filename': '27_243_45.2.bin',
    },
}


def load_poseformerv2_model(
    variant: str = '3-27-47.9',
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None,
    download_if_missing: bool = True
) -> nn.Module:
    """
    Load PoseFormerV2 model with pre-trained weights.

    Args:
        variant: Model variant (e.g., '3-27-47.9', '27-243-45.2')
        device: Device to load on
        checkpoint_path: Path to checkpoint file (None to auto-download)
        download_if_missing: Whether to download weights if not found

    Returns:
        Loaded PoseFormerV2 model in eval mode
    """
    if not POSEFORMERV2_AVAILABLE:
        raise ImportError(
            "PoseFormerV2 model not available. "
            "Make sure the repository is cloned at models/poseformerv2/"
        )

    if variant not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    # Get model configuration
    config = MODEL_CONFIGS[variant]

    # Create model
    logger.info(f"Creating PoseFormerV2 model (variant: {variant})...")
    model = PoseTransformerV2(
        num_frame=config['num_frame'],
        num_joints=17,  # COCO-17 format
        in_chans=2,     # x, y coordinates
        embed_dim_ratio=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=config['qkv_bias'],
        qk_scale=config['qk_scale'],
        drop_rate=config['drop_rate'],
        attn_drop_rate=config['attn_drop_rate'],
        drop_path_rate=config['drop_path_rate'],
        frame_kept=config['frame_kept'],
        coeff_kept=config['coeff_kept'],
    )

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoint" / "poseformerv2"
        checkpoint_path = checkpoint_dir / PRETRAINED_WEIGHTS[variant]['filename']

    if not Path(checkpoint_path).exists():
        if download_if_missing:
            logger.warning(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"To download, run: bash scripts/download_poseformerv2_weights.sh"
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model state dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_pos' in checkpoint:
                state_dict = checkpoint['model_pos']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Successfully loaded PoseFormerV2 ({variant}) weights")
    else:
        logger.warning(
            f"Running without pre-trained weights. Model will need training or "
            f"download weights with: bash scripts/download_poseformerv2_weights.sh"
        )

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model


def get_model_config(variant: str) -> Dict:
    """
    Get configuration dictionary for a model variant.

    Args:
        variant: Model variant (e.g., '3-27-47.9')

    Returns:
        Configuration dictionary
    """
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")

    return MODEL_CONFIGS[variant].copy()


def get_variant_from_params(num_frames: int, frame_kept: int = 3) -> str:
    """
    Get the best matching variant for given parameters.

    Args:
        num_frames: Total number of frames (27, 81, or 243)
        frame_kept: Number of frames to keep (1, 3, 9, 27)

    Returns:
        Variant string
    """
    # Find matching variant
    for variant, config in MODEL_CONFIGS.items():
        if config['num_frame'] == num_frames and config['frame_kept'] == frame_kept:
            return variant

    # Fallback to closest match
    if num_frames <= 27:
        return '3-27-47.9'
    elif num_frames <= 81:
        return '3-81-47.1'
    else:
        return '27-243-45.2'


def download_weights(variant: str, output_dir: Optional[Path] = None):
    """
    Download pre-trained weights for a specific variant.

    Args:
        variant: Model variant (e.g., '3-27-47.9')
        output_dir: Directory to save weights (default: checkpoint/poseformerv2)
    """
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown required for downloading. Install with: pip install gdown")

    if variant not in PRETRAINED_WEIGHTS:
        raise ValueError(f"Unknown variant: {variant}")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "checkpoint" / "poseformerv2"

    output_dir.mkdir(parents=True, exist_ok=True)

    weight_info = PRETRAINED_WEIGHTS[variant]
    output_path = output_dir / weight_info['filename']

    if output_path.exists():
        logger.info(f"Weights already exist: {output_path}")
        return output_path

    logger.info(f"Downloading PoseFormerV2 ({variant}) weights...")
    gdown.download(
        id=weight_info['gdown_id'],
        output=str(output_path),
        quiet=False
    )

    logger.info(f"Downloaded to: {output_path}")
    return output_path


def list_available_variants() -> Dict[str, Dict]:
    """
    List all available model variants with their configurations.

    Returns:
        Dictionary mapping variant names to their configs
    """
    variants_info = {}
    for variant, config in MODEL_CONFIGS.items():
        mpjpe = variant.split('-')[-1]
        variants_info[variant] = {
            'num_frames': config['num_frame'],
            'frame_kept': config['frame_kept'],
            'coeff_kept': config['coeff_kept'],
            'mpjpe': f"{mpjpe} mm",
            'fps_estimate': f"{150 if config['num_frame'] <= 27 else 100 if config['num_frame'] <= 81 else 50}-{400 if config['num_frame'] <= 27 else 300 if config['num_frame'] <= 81 else 200}",
            'best_for': 'Real-time' if config['num_frame'] <= 27 else 'Balanced' if config['num_frame'] <= 81 else 'High accuracy',
        }
    return variants_info
