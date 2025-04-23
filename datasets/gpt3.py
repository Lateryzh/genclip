import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random

'''
version:
    2去掉分布式
    3改成有监督的训练
    4融合多模态
'''


class MultiRegionDataset(Dataset):
    def __init__(self, root_dir, annotation_csv, transform=None):
        """
        Args:
            root_dir (str): 图像根目录
            annotation_csv (str): 标注文件路径，格式：
                patient_id,image_path,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,confidence,age,gender
            transform (callable): 同步变换图像和关键点
        """
        try:
            self.df = pd.read_csv(annotation_csv)
            # 按image_path分组（每个image可能对应多个区域）
            self.grouped = self.df.groupby('image_path')
            print(f"成功加载标注文件，共 {len(self.grouped)} 组数据")
            # print('datasets/gpt3.py:33', self.grouped)
        except Exception as e:
            raise RuntimeError(f"加载标注文件失败: {str(e)}")

        self.root = root_dir
        self.transform = transform

        # 元数据预处理配置
        self.age_bins = [0, 18, 30, 45, 60, 100]
        self.gender_map = {'male': 0, 'female': 1}

        # 新增数据验证
        valid_samples = []
        for img_path in self.grouped.groups.keys():
            # print("datasets/gpt3.py:46 img_path: ",img_path)
            full_path = os.path.join(root_dir, img_path)
            if not os.path.exists(full_path):
                print(f"警告：跳过无效图像路径 {full_path}")
                continue
            valid_samples.append(img_path)
        self.grouped = self.df[self.df['image_path'].isin(valid_samples)].groupby('image_path')
        print(f"有效样本数量: {len(self.grouped)}")

    def __len__(self):
        return len(self.grouped)

    def _process_age(self, age):
        """将年龄离散化为one-hot编码"""
        age_bin = np.digitize(age, self.age_bins) - 1
        return torch.nn.functional.one_hot(
            torch.tensor(age_bin),
            num_classes=len(self.age_bins)
        ).float()

    def _process_gender(self, gender):
        """将性别映射为整数"""
        return torch.tensor(
            self.gender_map.get(str(gender).lower(), -1),
            dtype=torch.long
        )

    def __getitem__(self, idx):
        # 获取分组数据（image_path作为键）
        (img_path, group) = list(self.grouped)[idx]  # group包含同一图像的所有区域数据

        # 构建完整文件路径
        full_path = os.path.join(self.root, img_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"图像文件不存在: {full_path}")
        image = Image.open(full_path).convert('RGB')

        # 解析关键点数据（每组17个区域，每个区域5个关键点）
        all_keypoints = []
        confidences = []
        for _, row in group.iterrows():
            # 从第3列开始取10个值（x1,y1到x5,y5）
            coords = np.array(row.iloc[2:12].tolist(), dtype=np.float32)

            # 转换为 (5,2) 坐标矩阵
            kps = coords.reshape(5, 2)
            all_keypoints.append(kps)

            # 获取置信度（第13列）
            confidences.append(float(row.iloc[12]))

        # 转换为numpy数组
        keypoints = np.stack(all_keypoints)  # Shape: (17,5,2)
        confidences = np.array(confidences, dtype=np.float32)  # Shape: (17,)

        # 获取元数据（假设同图像数据共享相同的元信息）
        age = group.iloc[0]['age']
        gender = group.iloc[0]['gender']

        # 应用数据增强变换
        if self.transform:
            vqgan_img, clip_img, keypoints = self.transform(image, keypoints)

        # 返回结构化数据
        return (
            vqgan_img,  # 预处理后的VQGAN输入图像 [C, H, W]
            clip_img,  # 预处理后的CLIP输入图像 [C, H, W]
            torch.from_numpy(keypoints).float(),  # 关键点坐标 [17,5,2]
            torch.from_numpy(confidences).float(),  # 关键点置信度 [17]
            self._process_age(age),  # 年龄的one-hot编码 [n_bins]
            self._process_gender(gender)  # 性别编码 [1]
        )


class PairedDataset(MultiRegionDataset):
    def __getitem__(self, idx):
        # 获取原始数据
        base_data = super().__getitem__(idx) # idx是一个整数
        # print("datasets/gpt3.py:123 len(self): ",len(self))
        # print("124 base_data: ", base_data)
        if len(self) < 2:
            raise ValueError("配对数据集需要至少2个样本")

        # 随机选择配对样本（确保与base不同）
        pair_idx = idx
        while pair_idx == idx:
            pair_idx = random.randint(0, len(self) - 1)

        pair_data = super().__getitem__(pair_idx)


        return {
            'vqgan_imgs': torch.stack((base_data[0], pair_data[0])),  # (base_img, pair_img)
            'clip_imgs': torch.stack((base_data[1], pair_data[1])),
            'keypoints': torch.stack([base_data[2], pair_data[2]]),
            'confidences': (base_data[3], pair_data[3]),
            'ages': base_data[4],  # 元数据取自基础样本
            'genders': base_data[5]
        }

class KeypointTransform:
    def __init__(self, split='train', vqgan_size=(256, 256), clip_size=(224, 224)):
        self.split = split
        self.vqgan_size = vqgan_size
        self.clip_size = clip_size


        # 基础几何变换（如翻转）
        if split == 'train':
            self.geom_transform = T.RandomHorizontalFlip(p=0.5)
        else:
            self.geom_transform = None

        # VQGAN的预处理流程
        self.vqgan_transform = T.Compose([
            T.Resize(vqgan_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # CLIP的预处理流程
        self.clip_transform = T.Compose([
            T.Resize(clip_size),
            T.CenterCrop(clip_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image, keypoints):
        # 应用几何变换（如水平翻转）
        if self.split == 'train' and self.geom_transform is not None:
            if torch.rand(1) < 0.5:
                image = T.functional.hflip(image)
                keypoints[:, :, 0] = image.width - keypoints[:, :, 0]

        # 保存原始尺寸以计算缩放比例
        # orig_w, orig_h = image.size
        # 坐标的那个模型输出是512*1024的大小
        orig_w = 512
        orig_h = 1024

        # 处理VQGAN图像
        vqgan_image = self.vqgan_transform(image)

        # 处理CLIP图像
        clip_image = self.clip_transform(image)

        # 调整关键点坐标到VQGAN尺寸
        scale_x = self.vqgan_size[0] / orig_w
        scale_y = self.vqgan_size[1] / orig_h
        keypoints[:, :, 0] *= scale_x
        keypoints[:, :, 1] *= scale_y

        return vqgan_image, clip_image, keypoints






