# MedAI-ModelZoo

欢迎来到 **MedAI-ModelZoo** 👏！本项目致力于提供医学图像分割领域的各种经典网络架构、即插即用的组件模块以及详尽的教程。

## 📦 已复现模型库 (Model Zoo)

| 模型名称 | 论文原标题 | 发表日期/出处 | 链接 | 2D简单测试 | 2D中等测试 | 3D简单测试 | 3D中等测试 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **U-Net** | *U-Net: Convolutional Networks for Biomedical Image Segmentation* | 2015-05 (MICCAI) | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | Dice=0.7487 (E48) | ✅ | / | / |
| **UNet++** | *UNet++: A Nested U-Net Architecture for Medical Image Segmentation* | 2018-07 (DLMIA) | [arXiv:1807.10165](https://arxiv.org/abs/1807.10165) | ✅ | ✅ | / | / |

- 2D简单测试使用 **DRIVE**，图像统一 `resize` 到 `512`。
- 2D中等测试使用 **Kvasir-SEG**，图像统一 `resize` 到 `352`。
- 3D简单测试使用 **MSD Task04 Hippocampus**，体数据统一 `resize` 到 `96`。
- 3D中等测试使用 **MSD Task09 Spleen**，体数据统一 `resize` 到 `96`。
- 训练配置采用通用策略：`optimizer=AdamW`、`loss=BCE + Dice`，并按任务类型分别设置 `batch_size` 与 `image_size`。
- 当前默认批次划分为：`twoD_simple_batch_size=4`、`twoD_medium_batch_size=4`、`threeD_simple_batch_size=1`、`threeD_medium_batch_size=1`（详见 `my_lib/test/train_config.py`）。

## 🚀 快速开始

（待补充：模块使用说明与安装步骤）