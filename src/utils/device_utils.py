"""
Device Utilities
Intelligent device detection and selection for CUDA, Apple Silicon MPS, and CPU.

Supports:
- NVIDIA GPUs (CUDA)
- Apple Silicon M1/M2/M3 (MPS - Metal Performance Shaders)
- CPU fallback

Author: SwimVision Pro Team
Date: 2025-01-20
"""

import torch
import platform
import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


class DeviceManager:
    """
    Manages device selection and provides utilities for cross-platform acceleration.

    Automatically detects and uses the best available hardware:
    1. NVIDIA GPU (CUDA) - if available
    2. Apple Silicon (MPS) - if available
    3. CPU - fallback
    """

    _instance = None
    _detected_device: Optional[str] = None

    def __new__(cls):
        """Singleton pattern to cache device detection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def detect_device() -> DeviceType:
        """
        Detect the best available device.

        Returns:
            Device type: "cuda", "mps", or "cpu"
        """
        # Check CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available: {gpu_name} ({gpu_count} GPU(s))")
            return device

        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            device = "mps"
            # Get system info
            system = platform.system()
            machine = platform.machine()
            processor = platform.processor()
            logger.info(f"✅ MPS available: {system} {machine} ({processor})")

            # Check if MPS is actually working
            try:
                # Test MPS with a simple operation
                test_tensor = torch.randn(10, 10, device="mps")
                _ = test_tensor @ test_tensor
                logger.info("✅ MPS backend verified and working")
            except Exception as e:
                logger.warning(f"⚠️ MPS available but not working: {e}")
                logger.warning("Falling back to CPU")
                return "cpu"

            return device

        # Fallback to CPU
        logger.info("ℹ️ No GPU acceleration available, using CPU")
        return "cpu"

    @staticmethod
    def get_device(preferred: Optional[str] = None) -> str:
        """
        Get the device to use, with optional preference.

        Args:
            preferred: Preferred device ("cuda", "mps", "cpu", or None for auto)

        Returns:
            Device string for PyTorch (e.g., "cuda:0", "mps", "cpu")
        """
        # Auto-detect if no preference
        if preferred is None or preferred == "auto":
            detected = DeviceManager.detect_device()
            return DeviceManager._format_device(detected)

        # Validate preferred device
        preferred_lower = preferred.lower()

        if preferred_lower in ["cuda", "gpu"]:
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                logger.warning("CUDA requested but not available. Falling back to auto-detect.")
                detected = DeviceManager.detect_device()
                return DeviceManager._format_device(detected)

        elif preferred_lower == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                logger.warning("MPS requested but not available. Falling back to auto-detect.")
                detected = DeviceManager.detect_device()
                return DeviceManager._format_device(detected)

        elif preferred_lower == "cpu":
            return "cpu"

        else:
            logger.warning(f"Unknown device '{preferred}'. Auto-detecting...")
            detected = DeviceManager.detect_device()
            return DeviceManager._format_device(detected)

    @staticmethod
    def _format_device(device_type: DeviceType) -> str:
        """Format device type to PyTorch device string."""
        if device_type == "cuda":
            return "cuda:0"
        else:
            return device_type  # "mps" or "cpu"

    @staticmethod
    def get_device_info() -> dict:
        """
        Get detailed information about available devices.

        Returns:
            Dictionary with device information
        """
        info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "recommended_device": DeviceManager.detect_device()
        }

        # CUDA details
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            info["gpu_memory"] = [
                f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                for i in range(torch.cuda.device_count())
            ]

        # MPS details
        if torch.backends.mps.is_available():
            info["mps_backend"] = "Metal Performance Shaders"
            # Try to get more Mac-specific info
            try:
                import subprocess
                # Get chip info (M1/M2/M3)
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    info["chip"] = result.stdout.strip()
            except:
                pass

        return info

    @staticmethod
    def print_device_info():
        """Print formatted device information."""
        info = DeviceManager.get_device_info()

        print("\n" + "="*60)
        print("DEVICE INFORMATION")
        print("="*60)
        print(f"Platform:          {info['platform']}")
        print(f"Machine:           {info['machine']}")
        print(f"Processor:         {info['processor']}")
        if 'chip' in info:
            print(f"Chip:              {info['chip']}")
        print(f"Python:            {info['python_version']}")
        print(f"PyTorch:           {info['pytorch_version']}")
        print()
        print(f"CUDA Available:    {info['cuda_available']}")
        if info['cuda_available']:
            print(f"  CUDA Version:    {info['cuda_version']}")
            print(f"  cuDNN Version:   {info['cudnn_version']}")
            print(f"  GPU Count:       {info['gpu_count']}")
            for i, (name, mem) in enumerate(zip(info['gpu_names'], info['gpu_memory'])):
                print(f"  GPU {i}:           {name} ({mem})")
        print()
        print(f"MPS Available:     {info['mps_available']}")
        if info['mps_available']:
            print(f"  Backend:         {info.get('mps_backend', 'N/A')}")
        print()
        print(f"Recommended:       {info['recommended_device'].upper()}")
        print("="*60 + "\n")

    @staticmethod
    def optimize_for_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
        """
        Optimize model for specific device.

        Args:
            model: PyTorch model
            device: Device string

        Returns:
            Optimized model
        """
        model = model.to(device)

        # Device-specific optimizations
        if "cuda" in device:
            # Enable cuDNN benchmarking for CUDA
            torch.backends.cudnn.benchmark = True
            logger.info("✅ Enabled cuDNN benchmarking for CUDA")

        elif device == "mps":
            # MPS-specific optimizations
            # Note: MPS is relatively new, optimization options are limited
            logger.info("✅ Model moved to MPS (Metal Performance Shaders)")

        else:  # CPU
            # CPU optimizations
            torch.set_num_threads(torch.get_num_threads())
            logger.info(f"✅ Using {torch.get_num_threads()} CPU threads")

        return model

    @staticmethod
    def move_to_device(data, device: str):
        """
        Move data (tensor, list, dict, etc.) to device.

        Handles nested structures like lists of tensors, dicts, etc.

        Args:
            data: Data to move (tensor, list, dict, etc.)
            device: Target device

        Returns:
            Data on the target device
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: DeviceManager.move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [DeviceManager.move_to_device(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(DeviceManager.move_to_device(item, device) for item in data)
        else:
            # Not a tensor, return as-is
            return data


def get_optimal_device(preferred: Optional[str] = None) -> str:
    """
    Convenience function to get optimal device.

    Args:
        preferred: Preferred device or None for auto-detection

    Returns:
        Device string for PyTorch
    """
    return DeviceManager.get_device(preferred)


def print_device_info():
    """Convenience function to print device info."""
    DeviceManager.print_device_info()


# Auto-detect on module import
_AUTO_DEVICE = DeviceManager.detect_device()
logger.info(f"Auto-detected device: {_AUTO_DEVICE}")


if __name__ == "__main__":
    # Demo device detection
    print_device_info()

    # Test device selection
    print("\nTesting device selection:")
    print(f"Auto:         {get_optimal_device()}")
    print(f"Preferred CUDA: {get_optimal_device('cuda')}")
    print(f"Preferred MPS:  {get_optimal_device('mps')}")
    print(f"Preferred CPU:  {get_optimal_device('cpu')}")

    # Test tensor operations
    print("\nTesting tensor operations on detected device:")
    device = get_optimal_device()
    try:
        x = torch.randn(100, 100, device=device)
        y = x @ x
        print(f"✅ Matrix multiplication successful on {device}")
        print(f"   Result shape: {y.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
