import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DATA_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_CACHE_ROOT = DEFAULT_LOCAL_DATA_ROOT  # 兼容旧变量名

DATASET_META: Dict[str, Dict[str, object]] = {
    "drive": {
        "archive_names": ["DRIVE.zip"],
        "folder_name": "DRIVE",
        "task": "task2_2d_simple",
        "display_name": "DRIVE",
        "display_level": "2D网络简单难度",
    },
    "kvasir_seg": {
        "archive_names": ["kvasir-seg.zip"],
        "folder_name": "Kvasir-SEG",
        "task": "task3_2d_medium",
        "display_name": "Kvasir-SEG",
        "display_level": "2D网络中等难度",
    },
    "msd_task04_hippocampus": {
        "archive_names": ["Task04_Hippocampus.tar"],
        "folder_name": "Task04_Hippocampus",
        "task": "task2_3d_simple",
        "display_name": "MSD Task04_Hippocampus",
        "display_level": "3D网络简单难度",
    },
    "msd_task09_spleen": {
        "archive_names": ["Task09_Spleen.tar"],
        "folder_name": "Task09_Spleen",
        "task": "task3_3d_medium",
        "display_name": "MSD Task09_Spleen",
        "display_level": "3D网络中等难度",
    },
}


def _extract_archive(archive_path: Path, dst_dir: Path) -> None:
    print(f"开始解压本地压缩包: {archive_path.name}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dst_dir)
    elif archive_path.suffix in {".tar", ".gz"} or archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dst_dir)
    else:
        raise ValueError(f"不支持的压缩格式: {archive_path}")
    print(f"✅ 解压完成: {archive_path.name}")


def _find_local_archive(meta: Dict[str, object], data_root: Path) -> Optional[Path]:
    archive_names = meta["archive_names"]
    if not isinstance(archive_names, list):
        raise TypeError("archive_names 必须为字符串列表")

    # 仅保留当前约定：压缩包放在 datasets 根目录
    for archive_name in archive_names:
        candidate = data_root / archive_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def ensure_local_dataset(dataset_key: str, data_root: Optional[Path] = None) -> Path:
    """确保本地数据集目录可用：存在则直接返回，不存在则从本地压缩包解压。"""
    if dataset_key not in DATASET_META:
        raise KeyError(f"未知数据集: {dataset_key}")

    meta = DATASET_META[dataset_key]
    root = data_root or DEFAULT_LOCAL_DATA_ROOT

    dataset_dir = root / str(meta["task"]) / str(meta["folder_name"])
    if dataset_dir.exists():
        print(f"✅ 本地数据集就绪: {dataset_dir}")
        return dataset_dir

    task_dir = root / str(meta["task"])
    task_dir.mkdir(parents=True, exist_ok=True)

    archive_path = _find_local_archive(meta, root)
    if archive_path is None:
        archive_list = ", ".join(str(x) for x in meta["archive_names"])
        raise FileNotFoundError(
            "未找到本地数据集压缩包。\n"
            f"数据集键: {dataset_key}\n"
            f"可接受文件名: {archive_list}\n"
            f"请放到: {root}"
        )

    print(f"正在准备{meta['display_level']}数据集（{meta['display_name']}），仅本地模式")
    _extract_archive(archive_path, task_dir)

    if not dataset_dir.exists():
        dirs = [p for p in task_dir.iterdir() if p.is_dir()]
        if len(dirs) == 1:
            dirs[0].rename(dataset_dir)

    if not dataset_dir.exists():
        raise RuntimeError(f"数据集解压后未找到目录: {dataset_dir}")

    print(f"✅ 数据集目录就绪: {dataset_dir}")
    return dataset_dir


def load_dataset_to_memory(dataset_key: str, data_root: Optional[Path] = None) -> Dict[str, object]:
    """加载本地测试数据集到内存，并显示 tqdm 进度条。"""
    dataset_dir = ensure_local_dataset(dataset_key, data_root=data_root)
    all_files = [p for p in dataset_dir.rglob("*") if p.is_file()]

    in_memory_files: Dict[str, bytes] = {}
    total_bytes = 0

    for fp in tqdm(all_files, desc=f"加载到内存: {dataset_key}", unit="file"):
        data = fp.read_bytes()
        rel = fp.relative_to(dataset_dir).as_posix()
        in_memory_files[rel] = data
        total_bytes += len(data)

    summary = {
        "dataset_key": dataset_key,
        "dataset_dir": str(dataset_dir),
        "num_files": len(all_files),
        "total_bytes": total_bytes,
    }

    print(
        f"✅ 内存加载完成: key={dataset_key}, "
        f"files={summary['num_files']}, total_bytes={summary['total_bytes']}"
    )

    return {"summary": summary, "files": in_memory_files}


def check_local_datasets(dataset_keys: Iterable[str], data_root: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """批量检查/准备本地数据集目录。"""
    result: Dict[str, Dict[str, str]] = {}
    for key in dataset_keys:
        try:
            path = ensure_local_dataset(key, data_root=data_root)
            result[key] = {"status": "ok", "path": str(path)}
        except Exception as exc:  # noqa: BLE001
            result[key] = {"status": "error", "error": str(exc)}
    return result


def load_default_test_datasets_to_memory(data_root: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    return {
        key: load_dataset_to_memory(key, data_root=data_root)
        for key in ["drive", "kvasir_seg", "msd_task04_hippocampus", "msd_task09_spleen"]
    }


def check_default_test_datasets(data_root: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    return check_local_datasets(
        dataset_keys=["drive", "kvasir_seg", "msd_task04_hippocampus", "msd_task09_spleen"],
        data_root=data_root,
    )
