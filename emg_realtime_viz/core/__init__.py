"""
Core modules for EMG processing
"""

from .data_loader import NinaproLoader
from .feature_extractor import FeatureExtractor
from .glove import glove_to_angles, normalize_glove, to_hand_angles
from .inference import TorchInference, energy_demo_model, load_torch_model, wave_demo_model
from .stream import DataStream

__all__ = [
    "NinaproLoader",
    "FeatureExtractor",
    "DataStream",
    "normalize_glove",
    "glove_to_angles",
    "to_hand_angles",
    "TorchInference",
    "load_torch_model",
    "energy_demo_model",
    "wave_demo_model",
]
