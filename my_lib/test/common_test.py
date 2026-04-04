from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from my_lib.test.load_test_dataset import ensure_local_dataset
from my_lib.test.train_config import (
    TrainConfig,
    config_to_dict,
    get_case_batch_size,
    get_case_image_size,
    get_train_config,
    override_config,
)

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 numpy: pip install numpy") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 Pillow: pip install Pillow") from exc


@dataclass(frozen=True)
class DatasetMeta:
    dataset_key: str
    spatial_dims: int
    in_channels: int
    classes: int


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，自动回退到 CPU")
        return torch.device("cpu")
    return torch.device(device)


def _read_2d_image(path: Path, is_mask: bool) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("L") if is_mask else img.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    if is_mask:
        return torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)
    return torch.from_numpy(arr / 255.0).permute(2, 0, 1)


def _resize_2d(t: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    align = False if mode != "nearest" else None
    return torch.nn.functional.interpolate(t.unsqueeze(0), size=(size, size), mode=mode, align_corners=align).squeeze(0)


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("3D 数据读取需要 nibabel，请安装: pip install nibabel") from exc
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _resize_3d(t: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    align = False if mode != "nearest" else None
    return torch.nn.functional.interpolate(t.unsqueeze(0), size=(size, size, size), mode=mode, align_corners=align).squeeze(0)


def _robust_zscore(x: torch.Tensor, q_low: float = 0.05, q_high: float = 0.95) -> torch.Tensor:
    flat = x.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    x = torch.clamp(x, min=lo, max=hi)
    return (x - x.mean()) / (x.std() + 1e-6)


def _roi_bbox(mask_2d: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = torch.where(mask_2d > 0)
    if ys.numel() == 0:
        return None
    y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
    x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
    return y0, y1, x0, x1


def _crop_2d(t: torch.Tensor, bbox: Optional[Tuple[int, int, int, int]]) -> torch.Tensor:
    if bbox is None:
        return t
    y0, y1, x0, x1 = bbox
    return t[:, y0:y1, x0:x1]


def _build_drive_samples(dataset_dir: Path) -> List[Tuple[Path, Path, Optional[Path]]]:
    images_root = dataset_dir / "training" / "images"
    masks_root = dataset_dir / "training" / "1st_manual"
    roi_root = dataset_dir / "training" / "mask"
    if not images_root.exists() or not masks_root.exists():
        images_root = dataset_dir / "images"
        masks_root = dataset_dir / "1st_manual"
        roi_root = dataset_dir / "mask"

    samples: List[Tuple[Path, Path, Optional[Path]]] = []
    for image_path in sorted(images_root.glob("*.tif")):
        prefix = image_path.stem.split("_")[0]
        manual_path = masks_root / f"{prefix}_manual1.gif"
        roi_path = roi_root / f"{prefix}_training_mask.gif"
        if manual_path.exists():
            samples.append((image_path, manual_path, roi_path if roi_path.exists() else None))
    return samples


def _build_kvasir_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    images_root = dataset_dir / "images"
    masks_root = dataset_dir / "masks"

    image_paths = sorted(images_root.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(images_root.glob("*"))

    pairs: List[Tuple[Path, Path]] = []
    for image_path in image_paths:
        mask_path = masks_root / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def _build_msd_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    images_root = dataset_dir / "imagesTr"
    labels_root = dataset_dir / "labelsTr"

    image_paths = sorted(images_root.glob("*.nii.gz"))
    if not image_paths:
        image_paths = sorted(images_root.glob("*.nii"))

    pairs: List[Tuple[Path, Path]] = []
    for image_path in image_paths:
        label_path = labels_root / image_path.name
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _infer_binary_classes(mask_tensor: torch.Tensor) -> int:
    return 1 if float(mask_tensor.max().item()) <= 1.0 else int(mask_tensor.max().item()) + 1


def _infer_2d_meta(dataset_key: str, image_path: Path, mask_path: Path) -> DatasetMeta:
    image = _read_2d_image(image_path, is_mask=False)
    mask = _read_2d_image(mask_path, is_mask=True)
    return DatasetMeta(dataset_key=dataset_key, spatial_dims=2, in_channels=int(image.shape[0]), classes=_infer_binary_classes(mask))


def _infer_3d_meta(dataset_key: str, image_path: Path, mask_path: Path) -> DatasetMeta:
    image_np = _load_nifti(image_path)
    mask_np = _load_nifti(mask_path)
    in_channels = 1 if image_np.ndim == 3 else int(image_np.shape[0])
    max_label = float(np.max(mask_np))
    classes = 1 if max_label <= 1.0 else int(max_label) + 1
    return DatasetMeta(dataset_key=dataset_key, spatial_dims=3, in_channels=in_channels, classes=classes)


class DriveDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, Path, Optional[Path]]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, manual_path, roi_path = self.samples[idx]
        image = _read_2d_image(image_path, is_mask=False)
        manual = _read_2d_image(manual_path, is_mask=True)

        if roi_path is not None:
            roi = _read_2d_image(roi_path, is_mask=True)
            image = image * roi
            manual = manual * roi
            bbox = _roi_bbox(roi[0])
            image = _crop_2d(image, bbox)
            manual = _crop_2d(manual, bbox)

        image = _resize_2d(image, self.image_size, mode="bilinear")
        manual = _resize_2d(manual, self.image_size, mode="nearest")
        image = _robust_zscore(image)
        return image, manual


class Seg2DDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], image_size: int) -> None:
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = _read_2d_image(image_path, is_mask=False)
        mask = _read_2d_image(mask_path, is_mask=True)

        image = _resize_2d(image, self.image_size, mode="bilinear")
        mask = _resize_2d(mask, self.image_size, mode="nearest")
        image = _robust_zscore(image)
        return image, mask


class Seg3DDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], image_size: int) -> None:
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.pairs[idx]
        image = torch.from_numpy(_load_nifti(image_path))
        label = torch.from_numpy((_load_nifti(label_path) > 0).astype(np.float32))

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if label.ndim == 3:
            label = label.unsqueeze(0)

        image = _resize_3d(image, self.image_size, mode="trilinear")
        label = _resize_3d(label, self.image_size, mode="nearest")
        image = _robust_zscore(image)
        return image, label


def infer_dataset_meta(dataset_key: str) -> DatasetMeta:
    dataset_dir = ensure_local_dataset(dataset_key)

    if dataset_key == "drive":
        samples = _build_drive_samples(dataset_dir)
        if not samples:
            raise RuntimeError(f"DRIVE 数据配对失败: {dataset_dir}")
        return _infer_2d_meta(dataset_key, samples[0][0], samples[0][1])

    if dataset_key == "kvasir_seg":
        pairs = _build_kvasir_pairs(dataset_dir)
        if not pairs:
            raise RuntimeError(f"Kvasir-SEG 数据配对失败: {dataset_dir}")
        return _infer_2d_meta(dataset_key, pairs[0][0], pairs[0][1])

    if dataset_key in {"msd_task04_hippocampus", "msd_task09_spleen"}:
        pairs = _build_msd_pairs(dataset_dir)
        if not pairs:
            raise RuntimeError(f"MSD 数据配对失败: {dataset_dir}")
        return _infer_3d_meta(dataset_key, pairs[0][0], pairs[0][1])

    raise KeyError(f"不支持的数据集键: {dataset_key}")


def _build_dataset_and_meta(dataset_key: str, cfg: TrainConfig) -> Tuple[Dataset, DatasetMeta, Path]:
    dataset_dir = ensure_local_dataset(dataset_key)
    image_size = get_case_image_size(cfg, dataset_key)

    if dataset_key == "drive":
        samples = _build_drive_samples(dataset_dir)
        if not samples:
            raise RuntimeError(f"DRIVE 数据配对失败: {dataset_dir}")
        meta = _infer_2d_meta(dataset_key, samples[0][0], samples[0][1])
        return DriveDataset(samples=samples, image_size=image_size), meta, dataset_dir

    if dataset_key == "kvasir_seg":
        pairs = _build_kvasir_pairs(dataset_dir)
        if not pairs:
            raise RuntimeError(f"Kvasir-SEG 数据配对失败: {dataset_dir}")
        meta = _infer_2d_meta(dataset_key, pairs[0][0], pairs[0][1])
        return Seg2DDataset(pairs=pairs, image_size=image_size), meta, dataset_dir

    if dataset_key in {"msd_task04_hippocampus", "msd_task09_spleen"}:
        pairs = _build_msd_pairs(dataset_dir)
        if not pairs:
            raise RuntimeError(f"MSD 数据配对失败: {dataset_dir}")
        meta = _infer_3d_meta(dataset_key, pairs[0][0], pairs[0][1])
        return Seg3DDataset(pairs=pairs, image_size=image_size), meta, dataset_dir

    raise KeyError(f"不支持的数据集键: {dataset_key}")


def _merge_config_with_meta(cfg: TrainConfig, meta: DatasetMeta, dataset_key: str) -> Dict[str, object]:
    merged = config_to_dict(cfg)
    merged.update(asdict(meta))
    merged["selected_batch_size"] = get_case_batch_size(cfg, dataset_key)
    merged["selected_image_size"] = get_case_image_size(cfg, dataset_key)
    return merged


def _split_dataset(dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    total = len(dataset)
    if total < 2:
        raise RuntimeError("数据样本数不足，至少需要2个样本用于训练/测试划分")

    train_len = int(total * train_ratio)
    train_len = min(max(train_len, 1), total - 1)
    test_len = total - train_len
    train_ds, test_ds = random_split(dataset, [train_len, test_len])
    return train_ds, test_ds, {"total": total, "train": train_len, "test": test_len}


def _mixed_ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, ce_weight: float, dice_weight: float) -> torch.Tensor:
    if logits.shape[1] == 1:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        reduce_dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dim=reduce_dims)
        union = probs.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
        dice_loss = 1.0 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
        return ce_weight * bce + dice_weight * dice_loss

    target_idx = target.squeeze(1).long()
    ce = torch.nn.functional.cross_entropy(logits, target_idx)
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target_idx, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    reduce_dims = tuple(range(2, probs.ndim))
    intersection = (probs * one_hot).sum(dim=reduce_dims)
    union = probs.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    dice_loss = 1.0 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    return ce_weight * ce + dice_weight * dice_loss


def _dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > 0.5).float()
        reduce_dims = tuple(range(1, pred.ndim))
        intersection = (pred * target).sum(dim=reduce_dims)
        union = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return float(dice.mean().item())

    pred_idx = torch.argmax(logits, dim=1)
    target_idx = target.squeeze(1).long()
    pred_fg = (pred_idx > 0).float()
    target_fg = (target_idx > 0).float()
    reduce_dims = tuple(range(1, pred_fg.ndim))
    intersection = (pred_fg * target_fg).sum(dim=reduce_dims)
    union = pred_fg.sum(dim=reduce_dims) + target_fg.sum(dim=reduce_dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return float(dice.mean().item())


def _build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if not cfg.scheduler:
        return None
    if cfg.scheduler.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.scheduler_t_max, 1))
    return None


def _unwrap_logits(pred) -> torch.Tensor:
    return pred[-1] if isinstance(pred, list) else pred


def _run_one_phase(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    train_mode: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    model.train() if train_mode else model.eval()
    phase = "train" if train_mode else "test"

    loss_sum, dice_sum, n_steps = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=phase, unit="batch", leave=True)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train_mode):
            pred = model(images)
            logits = _unwrap_logits(pred)
            loss = _mixed_ce_dice_loss(logits, masks, cfg.ce_weight, cfg.dice_weight)
            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_v = float(loss.item())
        dice_v = _dice_score(logits.detach(), masks)
        loss_sum += loss_v
        dice_sum += dice_v
        n_steps += 1
        pbar.set_postfix({"loss": f"{loss_v:.4f}", "dice": f"{dice_v:.4f}"})

    if n_steps == 0:
        return {"loss": 0.0, "dice": 0.0}
    return {"loss": loss_sum / n_steps, "dice": dice_sum / n_steps}


def train_with_dataset_key(
    *,
    model: torch.nn.Module,
    dataset_key: str,
    dataset_name: str,
    device: str = "cuda",
    epochs: Optional[int] = None,
    train_ratio: Optional[float] = None,
) -> Dict[str, object]:
    cfg = override_config(get_train_config(dataset_key), epochs=epochs, train_ratio=train_ratio)

    full_dataset, meta, dataset_dir = _build_dataset_and_meta(dataset_key, cfg)
    merged_config = _merge_config_with_meta(cfg, meta, dataset_key)
    selected_batch_size = get_case_batch_size(cfg, dataset_key)

    train_ds, test_ds, split_info = _split_dataset(full_dataset, cfg.train_ratio)
    train_loader = DataLoader(train_ds, batch_size=selected_batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=selected_batch_size, shuffle=False, num_workers=cfg.num_workers)

    dev = resolve_device(device)
    model = model.to(dev)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    history: List[Dict[str, float]] = []

    for epoch_idx in range(cfg.epochs):
        print(f"\n[{dataset_name}] Epoch {epoch_idx + 1}/{cfg.epochs}")

        train_metrics = _run_one_phase(model=model, loader=train_loader, device=dev, cfg=cfg, train_mode=True, optimizer=optimizer)
        test_metrics = _run_one_phase(model=model, loader=test_loader, device=dev, cfg=cfg, train_mode=False)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch_idx + 1,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "test_loss": test_metrics["loss"],
            "test_dice": test_metrics["dice"],
        }
        history.append(row)

        print(
            f"Epoch {row['epoch']} | "
            f"train_loss={row['train_loss']:.4f}, train_dice={row['train_dice']:.4f}, "
            f"test_loss={row['test_loss']:.4f}, test_dice={row['test_dice']:.4f}"
        )

    return {
        "dataset": dataset_name,
        "dataset_key": dataset_key,
        "dataset_dir": str(dataset_dir),
        "split": split_info,
        "config": merged_config,
        "history": history,
    }


def train_drive(model: torch.nn.Module, device: str = "cuda", epochs: Optional[int] = None) -> Dict[str, object]:
    return train_with_dataset_key(model=model, dataset_key="drive", dataset_name="DRIVE", device=device, epochs=epochs)


def train_kvasir_seg(model: torch.nn.Module, device: str = "cuda", epochs: Optional[int] = None) -> Dict[str, object]:
    return train_with_dataset_key(model=model, dataset_key="kvasir_seg", dataset_name="Kvasir-SEG", device=device, epochs=epochs)


def train_task04_hippocampus(model: torch.nn.Module, device: str = "cuda", epochs: Optional[int] = None) -> Dict[str, object]:
    return train_with_dataset_key(
        model=model,
        dataset_key="msd_task04_hippocampus",
        dataset_name="MSD Task04_Hippocampus",
        device=device,
        epochs=epochs,
    )


def train_task09_spleen(model: torch.nn.Module, device: str = "cuda", epochs: Optional[int] = None) -> Dict[str, object]:
    return train_with_dataset_key(
        model=model,
        dataset_key="msd_task09_spleen",
        dataset_name="MSD Task09_Spleen",
        device=device,
        epochs=epochs,
    )
