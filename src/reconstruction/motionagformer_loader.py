"""
MotionAGFormer Model Loader
============================

Utility functions to load MotionAGFormer models from the repository.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from easydict import EasyDict

logger = logging.getLogger(__name__)

# Add MotionAGFormer repo to path
MOTIONAGFORMER_PATH = Path(__file__).parent.parent.parent / "models" / "motionagformer"
sys.path.insert(0, str(MOTIONAGFORMER_PATH))

# Import MotionAGFormer model
try:
    from model.MotionAGFormer import MotionAGFormer
    MOTIONAGFORMER_AVAILABLE = True
except ImportError:
    logger.warning(f"Failed to import MotionAGFormer from {MOTIONAGFORMER_PATH}")
    MOTIONAGFORMER_AVAILABLE = False


# Model configurations matching the YAML files
MODEL_CONFIGS = {
    'xs': {
        'n_layers': 12,
        'dim_in': 3,
        'dim_feat': 64,
        'dim_rep': 512,
        'dim_out': 3,
        'n_frames': 27,
        'num_joints': 17,
        'mlp_ratio': 4,
        'num_heads': 8,
        'qkv_bias': False,
        'qkv_scale': None,
        'use_layer_scale': True,
        'layer_scale_init_value': 0.00001,
        'use_adaptive_fusion': True,
        'hierarchical': False,
        'use_temporal_similarity': True,
        'neighbour_num': 2,
        'temporal_connection_len': 1,
        'use_tcn': False,
        'graph_only': False,
        'attn_drop': 0.0,
        'drop': 0.0,
        'drop_path': 0.0,
        'act_layer': 'gelu',
    },
    's': {
        'n_layers': 12,
        'dim_in': 3,
        'dim_feat': 96,
        'dim_rep': 512,
        'dim_out': 3,
        'n_frames': 81,
        'num_joints': 17,
        'mlp_ratio': 4,
        'num_heads': 8,
        'qkv_bias': False,
        'qkv_scale': None,
        'use_layer_scale': True,
        'layer_scale_init_value': 0.00001,
        'use_adaptive_fusion': True,
        'hierarchical': False,
        'use_temporal_similarity': True,
        'neighbour_num': 2,
        'temporal_connection_len': 1,
        'use_tcn': False,
        'graph_only': False,
        'attn_drop': 0.0,
        'drop': 0.0,
        'drop_path': 0.0,
        'act_layer': 'gelu',
    },
    'b': {
        'n_layers': 16,
        'dim_in': 3,
        'dim_feat': 128,
        'dim_rep': 512,
        'dim_out': 3,
        'n_frames': 243,
        'num_joints': 17,
        'mlp_ratio': 4,
        'num_heads': 8,
        'qkv_bias': False,
        'qkv_scale': None,
        'use_layer_scale': True,
        'layer_scale_init_value': 0.00001,
        'use_adaptive_fusion': True,
        'hierarchical': False,
        'use_temporal_similarity': True,
        'neighbour_num': 2,
        'temporal_connection_len': 1,
        'use_tcn': False,
        'graph_only': False,
        'attn_drop': 0.0,
        'drop': 0.0,
        'drop_path': 0.0,
        'act_layer': 'gelu',
    },
    'l': {
        'n_layers': 20,
        'dim_in': 3,
        'dim_feat': 160,
        'dim_rep': 512,
        'dim_out': 3,
        'n_frames': 243,
        'num_joints': 17,
        'mlp_ratio': 4,
        'num_heads': 8,
        'qkv_bias': False,
        'qkv_scale': None,
        'use_layer_scale': True,
        'layer_scale_init_value': 0.00001,
        'use_adaptive_fusion': True,
        'hierarchical': False,
        'use_temporal_similarity': True,
        'neighbour_num': 2,
        'temporal_connection_len': 1,
        'use_tcn': False,
        'graph_only': False,
        'attn_drop': 0.0,
        'drop': 0.0,
        'drop_path': 0.0,
        'act_layer': 'gelu',
    },
}

# Pre-trained weights paths and download info
PRETRAINED_WEIGHTS = {
    'xs': {
        'h36m': {
            'gdown_id': '1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH',
            'filename': 'motionagformer-xs-h36m.pth.tr',
        }
    },
    's': {
        'h36m': {
            'gdown_id': '1DrF7WZdDvRPsH12gQm5DPXbviZ4waYFf',
            'filename': 'motionagformer-s-h36m.pth.tr',
        }
    },
    'b': {
        'h36m': {
            'gdown_id': '1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP',
            'filename': 'motionagformer-b-h36m.pth.tr',
        }
    },
    'l': {
        'h36m': {
            'gdown_id': '1WI8QSsD84wlXIdK1dLp6hPZq4FPozmVZ',
            'filename': 'motionagformer-l-h36m.pth.tr',
        }
    },
}


def load_motionagformer_model(
    variant: str = 'xs',
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None,
    download_if_missing: bool = True
) -> nn.Module:
    """
    Load MotionAGFormer model with pre-trained weights.

    Args:
        variant: Model variant ('xs', 's', 'b', 'l')
        device: Device to load on
        checkpoint_path: Path to checkpoint file (None to auto-download)
        download_if_missing: Whether to download weights if not found

    Returns:
        Loaded MotionAGFormer model in eval mode
    """
    if not MOTIONAGFORMER_AVAILABLE:
        raise ImportError(
            "MotionAGFormer model not available. "
            "Make sure the repository is cloned at models/motionagformer/"
        )

    variant = variant.lower()
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}")

    # Get model configuration
    config = EasyDict(MODEL_CONFIGS[variant])

    # Create model
    logger.info(f"Creating MotionAGFormer-{variant.upper()} model...")
    model = MotionAGFormer(
        n_layers=config.n_layers,
        dim_in=config.dim_in,
        dim_feat=config.dim_feat,
        dim_rep=config.dim_rep,
        dim_out=config.dim_out,
        mlp_ratio=config.mlp_ratio,
        act_layer=config.act_layer,
        attn_drop=config.attn_drop,
        drop=config.drop,
        drop_path=config.drop_path,
        use_layer_scale=config.use_layer_scale,
        layer_scale_init_value=config.layer_scale_init_value,
        use_adaptive_fusion=config.use_adaptive_fusion,
        num_heads=config.num_heads,
        qkv_bias=config.qkv_bias,
        qkv_scale=config.qkv_scale,
        hierarchical=config.hierarchical,
        use_temporal_similarity=config.use_temporal_similarity,
        temporal_connection_len=config.temporal_connection_len,
        use_tcn=config.use_tcn,
        graph_only=config.graph_only,
        neighbour_num=config.neighbour_num,
        n_frames=config.n_frames,
    )

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoint" / "motionagformer"
        checkpoint_path = checkpoint_dir / PRETRAINED_WEIGHTS[variant]['h36m']['filename']

    if not Path(checkpoint_path).exists():
        if download_if_missing:
            logger.warning(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"To download, run: bash scripts/download_motionagformer_weights.sh"
            )
            # Could implement auto-download here with gdown
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model state dict (checkpoint may contain optimizer, etc.)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Successfully loaded MotionAGFormer-{variant.upper()} weights")
    else:
        logger.warning(
            f"Running without pre-trained weights. Model will need training or "
            f"download weights with: bash scripts/download_motionagformer_weights.sh"
        )

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model


def get_model_config(variant: str) -> Dict:
    """
    Get configuration dictionary for a model variant.

    Args:
        variant: Model variant ('xs', 's', 'b', 'l')

    Returns:
        Configuration dictionary
    """
    variant = variant.lower()
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")

    return MODEL_CONFIGS[variant].copy()


def download_weights(variant: str, output_dir: Optional[Path] = None):
    """
    Download pre-trained weights for a specific variant.

    Args:
        variant: Model variant ('xs', 's', 'b', 'l')
        output_dir: Directory to save weights (default: checkpoint/motionagformer)
    """
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown required for downloading. Install with: pip install gdown")

    variant = variant.lower()
    if variant not in PRETRAINED_WEIGHTS:
        raise ValueError(f"Unknown variant: {variant}")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "checkpoint" / "motionagformer"

    output_dir.mkdir(parents=True, exist_ok=True)

    weight_info = PRETRAINED_WEIGHTS[variant]['h36m']
    output_path = output_dir / weight_info['filename']

    if output_path.exists():
        logger.info(f"Weights already exist: {output_path}")
        return output_path

    logger.info(f"Downloading MotionAGFormer-{variant.upper()} weights...")
    gdown.download(
        id=weight_info['gdown_id'],
        output=str(output_path),
        quiet=False
    )

    logger.info(f"Downloaded to: {output_path}")
    return output_path
