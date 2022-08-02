from typing import Optional, List
from simple_parsing import ArgumentParser
from dataclasses import dataclass, field
from simple_parsing.helpers import list_field

# ======================
# LOGGING
# ======================
@dataclass
class Logging:
    project: str = "gaussflow"
    entity: str = "ige"
    log_dir: str = "/mnt/meom/workdir/johnsonj/logs"
    resume: str = "allow"
    mode: str = "disabled"
    smoke_test: str = "store_true"
    id: Optional[str] = None
    run_path: Optional[str] = None
    model_path: Optional[str] = None


# ======================
# TRAIN/VAL SPLIT
# ======================
@dataclass
class TrainTestSplit:
    # split sizes
    n_train: int = 10_000
    n_valid: int = 2_000
    n_test: int = 10_000

    # noise
    noise_train: float = 0.05

    # randomnewss
    seed: int = 42
    seed_valid: int = 123
    seed_test: int = 666

# ======================
# DATALOADER
# ======================
@dataclass
class DataLoader:
    # dataloader
    train_shuffle: bool = True
    pin_memory: bool = False
    num_workers: int = 0
    batch_size: int = 4096
    batch_size_eval: int = 10_000

# ======================
# MODEL
# ======================
@dataclass
class Model:
    model: str = "gflow" # Options: "gflow", "flow", "nsf", "flowpp", "custom"

# ======================
# Flow Arguments
# ======================
@dataclass
class Flow:
    # model specific
    num_layers: int = 12
    # marginal gaussianization
    mg_layer: str = "splinerq"
    num_mixtures: int = 8
    # orthogonal layer
    fast_householder: bool = True
    num_reflections: int = 2

# ======================
# LOSSES
# ======================
@dataclass
class Losses:
    loss: str = "nll"  # Options: "nll", "kld", "iw"


# ======================
# OPTIMIZER
# ======================
@dataclass
class Optimizer:
    optimizer: str = "adam"  # Options: "adam", "adamw" # "adamax"
    learning_rate: float = 1e-4
    num_epochs: int = 300
    min_epochs: int = 1
    device: str = "cpu"
    gpus: int = 0  # the number of GPUS (pytorch-lightning)
    mps: bool = False # Specific for Macbook

# ======================
# LR Scheduler
# ======================
@dataclass
class LRScheduler:
    # LR Scheduler
    lr_scheduler: str = "warmcosanneal"  # Options: "cosanneal", "exp"

    # Specific: "warmcosanneal"
    warmup: int = 10
    min_lr: float = 0.0

    # STEPLR Specific
    milestones: List[int] = list_field(500, 1000, 1500, 2000, 2500)

# ======================
# Evaluation METRICS
# ======================
@dataclass
class Metrics:
    # binning along track
    n_samples: int = 10