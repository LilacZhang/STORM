import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
from einops import repeat
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN


class DeviceManager:
    """
    Manages device selection and AMP configuration for cross-platform training.
    Supports automatic detection: MPS > CUDA > CPU
    """

    def __init__(self, device_config="auto", use_amp=None):
        """
        Args:
            device_config: "auto", "mps", "cuda", "cpu"
            use_amp: True/False/None (None means auto-detect based on device)
        """
        self.device = self._detect_device(device_config)
        self.use_amp = self._determine_amp(use_amp)
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

    def _detect_device(self, device_config):
        """Detect and return the appropriate device"""
        if device_config == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                print(f"🚀 Using MPS device (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print(f"⚠️  Using CPU device (no GPU available)")
        elif device_config == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                print(f"🚀 Using MPS device (Apple Silicon GPU)")
            else:
                raise RuntimeError("MPS device requested but not available")
        elif device_config == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("CUDA device requested but not available")
        elif device_config == "cpu":
            device = torch.device("cpu")
            print(f"Using CPU device")
        else:
            raise ValueError(f"Unknown device config: {device_config}")

        return device

    def _determine_amp(self, use_amp):
        """Determine whether to use automatic mixed precision"""
        if use_amp is not None:
            # User explicitly specified
            if use_amp and self.device.type == "mps":
                print("⚠️  AMP requested but disabled on MPS (limited support)")
                return False
            return use_amp
        else:
            # Auto-detect: enable only on CUDA
            if self.device.type == "cuda":
                print("✓ AMP enabled (bfloat16)")
                return True
            else:
                print(f"✓ AMP disabled on {self.device.type} (using float32)")
                return False

    def get_device(self):
        """Return the torch device object"""
        return self.device

    def get_amp_dtype(self):
        """Return the appropriate dtype for AMP"""
        return self.tensor_dtype

    def create_grad_scaler(self):
        """Create GradScaler conditionally based on device and AMP setting"""
        if self.device.type == "cuda" and self.use_amp:
            return torch.cuda.amp.GradScaler(enabled=True)
        else:
            # Return a dummy scaler for compatibility
            return torch.cuda.amp.GradScaler(enabled=False)


def seed_np_torch(seed=20010105, device=None):
    """
    Set random seeds for reproducibility across numpy, random, and torch.

    Args:
        seed: Random seed value
        device: torch.device object or None (for backward compatibility)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Conditionally set CUDA seeds
    if device is None or device.type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Set MPS seeds if applicable
    if device is not None and device.type == "mps":
        # MPS uses the same manual_seed as CPU
        torch.manual_seed(seed)


class Logger:
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar:
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.Device = "auto"  # "auto", "mps", "cuda", "cpu"
    conf.BasicSettings.UseAMP = None  # None (auto), True, or False
    conf.BasicSettings.ReplayBufferOnGPU = False  # Deprecated, use Device instead

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
