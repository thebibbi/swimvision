"""
SwimVision Pipeline Module
Orchestrates all components for unified video processing.
"""

from src.pipeline.orchestrator import (
    FrameResult,
    PipelineConfig,
    ProcessingMode,
    SwimVisionPipeline,
)

__all__ = ["SwimVisionPipeline", "PipelineConfig", "ProcessingMode", "FrameResult"]
