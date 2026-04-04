from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional


@dataclass(frozen=True)
class TrainConfig:
    # training loop
    train_ratio: float = 0.8
    epochs: int = 50
    num_workers: int = 4

    # per-test-case batch size (4 cases)
    twoD_simple_batch_size: int = 2
    twoD_medium_batch_size: int = 4
    threeD_simple_batch_size: int = 1
    threeD_medium_batch_size: int = 1

    # per-test-case resize target (4 cases)
    twoD_simple_image_size: int = 512
    twoD_medium_image_size: int = 352
    threeD_simple_image_size: int = 96
    threeD_medium_image_size: int = 96

    # optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # scheduler
    scheduler: Optional[str] = "cosine"
    scheduler_t_max: int = 20

    # loss weights (CE + Dice)
    ce_weight: float = 0.5
    dice_weight: float = 0.5


DEFAULT_TRAIN_CONFIG = TrainConfig()


DATASET_CONFIGS: Dict[str, TrainConfig] = {
    "drive": DEFAULT_TRAIN_CONFIG,
    "kvasir_seg": DEFAULT_TRAIN_CONFIG,
    "msd_task04_hippocampus": DEFAULT_TRAIN_CONFIG,
    "msd_task09_spleen": DEFAULT_TRAIN_CONFIG,
}


DATASET_CASE_MAP: Dict[str, str] = {
    "drive": "twoD_simple",
    "kvasir_seg": "twoD_medium",
    "msd_task04_hippocampus": "threeD_simple",
    "msd_task09_spleen": "threeD_medium",
}


def get_train_config(dataset_key: str) -> TrainConfig:
    return DATASET_CONFIGS.get(dataset_key, DEFAULT_TRAIN_CONFIG)


def get_case_name(dataset_key: str) -> str:
    if dataset_key not in DATASET_CASE_MAP:
        raise KeyError(f"不支持的数据集键: {dataset_key}")
    return DATASET_CASE_MAP[dataset_key]


def get_case_batch_size(cfg: TrainConfig, dataset_key: str) -> int:
    case_name = get_case_name(dataset_key)
    return int(getattr(cfg, f"{case_name}_batch_size"))


def get_case_image_size(cfg: TrainConfig, dataset_key: str) -> int:
    case_name = get_case_name(dataset_key)
    return int(getattr(cfg, f"{case_name}_image_size"))


def override_config(
    cfg: TrainConfig,
    *,
    epochs: Optional[int] = None,
    train_ratio: Optional[float] = None,
) -> TrainConfig:
    updates = {}
    if epochs is not None:
        updates["epochs"] = epochs
        updates["scheduler_t_max"] = max(epochs, 1)
    if train_ratio is not None:
        updates["train_ratio"] = train_ratio
    return replace(cfg, **updates) if updates else cfg


def config_to_dict(cfg: TrainConfig) -> Dict[str, object]:
    return asdict(cfg)
