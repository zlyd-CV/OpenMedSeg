import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from my_lib.models.segmentors import UNet
from my_lib.test.common_test import infer_dataset_meta, resolve_device, train_drive, train_kvasir_seg
from my_lib.test.load_test_dataset import DEFAULT_CACHE_ROOT, check_local_datasets


def _build_unet(in_channels: int, out_classes: int) -> torch.nn.Module:
    num_classes = 1 if out_classes <= 2 else out_classes
    return UNet(in_channels=in_channels, num_classes=num_classes)


def test_unet(
    model: Optional[torch.nn.Module] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: str = "cuda",
) -> bool:
    if model is None:
        meta = infer_dataset_meta("drive")
        model = _build_unet(in_channels=meta.in_channels, out_classes=meta.classes)
    dev = resolve_device(device)
    model = model.to(dev).eval()

    x = torch.randn(*input_shape, device=dev)
    with torch.no_grad():
        y = model(x)

    print(f"✅ UNet 前向成功: input={tuple(x.shape)}, output={tuple(y.shape)}")
    return True


def test_twoDmodel_simple(
    model: Optional[torch.nn.Module] = None,
    epochs: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, object]:
    if model is None:
        meta = infer_dataset_meta("drive")
        model = _build_unet(in_channels=meta.in_channels, out_classes=meta.classes)
    return train_drive(model=model, device=device, epochs=epochs)


def test_twoDmodel_medium(
    model: Optional[torch.nn.Module] = None,
    epochs: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, object]:
    if model is None:
        meta = infer_dataset_meta("kvasir_seg")
        model = _build_unet(in_channels=meta.in_channels, out_classes=meta.classes)
    return train_kvasir_seg(model=model, device=device, epochs=epochs)


if __name__ == "__main__":
    print("=" * 80)
    print("UNet 测试入口（2D）")
    print(f"默认缓存根目录: {DEFAULT_CACHE_ROOT}")

    dataset_keys_to_check = ["drive", "kvasir_seg"]
    check_result = check_local_datasets(dataset_keys_to_check)
    print("2D数据集本地检查结果:")
    for key, info in check_result.items():
        print(f"  - {key}: {info}")

    test_unet(input_shape=(1, 3, 256, 256), device="cuda")
    test_twoDmodel_simple(device="cuda")
    test_twoDmodel_medium(device="cuda")

    print("UNet 2D演示流程已完成。")
