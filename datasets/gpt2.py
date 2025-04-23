import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

''''
数据集结构
data
    datasetname
        images
            xxx.png
    annotations: 标注文件路径，格式：
        train.csv
            image_path, x1,y1,...,x5,y5,confidence, age, gender
            xxx.png,x1...............................................
        val.csv

'''
class MultiRegionDataset(Dataset):
    def __init__(self, root_dir, annotation_csv, transform=None):
        """
        Args:
            root_dir (str): 图像根目录
            annotation_csv (str): 标注文件路径，格式：
                image_path, x1,y1,...,x5,y5,confidence, age, gender
            transform (callable): 同步变换图像和关键点
        """
        try:
            self.df = pd.read_csv(annotation_csv)
            # 按图像分组（假设每图有17个区域）
            self.grouped = self.df.groupby('image_path')
            print(f"成功加载标注文件，共 {len(self.grouped)} 组数据")
        except Exception as e:
            raise RuntimeError(f"加载标注文件失败: {str(e)}")

        self.root = root_dir
        # self.df = pd.read_csv(annotation_csv)
        self.transform = transform

        # 元数据预处理
        self.age_bins = [0, 18, 30, 45, 60, 100]
        self.gender_map = {'male': 0, 'female': 1}


        # 清洗非法值 ↓↓↓
        valid_genders = ['male', 'female']
        self.df = self.df[self.df['gender'].str.lower().isin(valid_genders)]


    def __len__(self):
        return len(self.grouped)
    def _process_age(self, age):
        age_bin = np.digitize(age, self.age_bins) - 1
        return torch.nn.functional.one_hot(
            torch.tensor(age_bin),
            num_classes=len(self.age_bins)
        ).float()

    def _process_gender(self, gender):
        return torch.tensor(
            self.gender_map.get(str(gender).lower(), -1),
            dtype=torch.float32
        )
    def __getitem__(self, idx):
        # 获取图像所有区域数据
        img_path, group = list(self.grouped)[idx]
        full_path = os.path.join(self.root, img_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"图像文件不存在: {full_path}")
        image = Image.open(full_path).convert('RGB')

        # 解析全部17个区域的关键点
        all_keypoints = []
        confidences = []
        for _, row in group.iterrows():
            # 解析坐标数据 (5点×2坐标 + 置信度)
            coords = np.array(row[2:-3].tolist(), dtype=np.float32)  # 排除路径、age、gender
            # 强制转换为浮点数
            confidence = float(row[-3])  # 添加类型转换

            # 转换为 (5,2) 坐标矩阵
            kps = coords.reshape(5, 2)
            all_keypoints.append(kps)
            confidences.append(confidence)

        # 转换为numpy数组
        keypoints = np.stack(all_keypoints)  # (17,5,2)
        # 转换为numpy数组时指定dtype
        confidences = np.array(confidences, dtype=np.float32)  # 确保是浮点类型

        # 元数据
        age = group.iloc[0]['age']
        gender = group.iloc[0]['gender']

        # 应用变换
        if self.transform:
            vqgan_img, clip_img, keypoints = self.transform(image, keypoints)

        # 元数据处理
        age_tensor = self._process_age(age)
        gender_tensor = self._process_gender(gender)

        # 确保所有返回值为张量且类型正确
        return (
            vqgan_img,  # 已为张量 [C, H, W]
            clip_img,  # 已为张量 [C, H, W]
            torch.from_numpy(keypoints).float(),  # 将numpy转为FloatTensor [17,5,2]
            torch.from_numpy(confidences).float(),  # 将numpy转为FloatTensor [17]
            age_tensor,  # 已是one-hot张量 [n_bins]
            gender_tensor.to(torch.long)  # 确保性别为整数类型 [1]
        )




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


# # ---
#
#
# def denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]):
#     # 反标准化处理
#     img = img_tensor.clone().permute(1, 2, 0).numpy()  # CxHxW -> HxWxC
#     img = img * std + mean
#     img = np.clip(img, 0, 1)
#     return img
#
#
# def plot_sample(sample, idx=0, n_regions=17, kps_per_region=5):
#     # 获取指定样本
#     img = sample['image'][idx]
#     keypoints = sample['keypoints'][idx].numpy()
#     print("keypoints: ", keypoints)
#     confidences = sample['confidences'][idx].numpy()
#
#     # 反标准化图像
#     img = denormalize(img)
#
#     # 创建画布
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img)
#
#     # 定义颜色循环（17个区域不同颜色）
#     colors = plt.cm.get_cmap('tab20', n_regions)
#
#     # 绘制每个区域的关键点
#     for region in range(n_regions):
#         # 跳过无数据区域（如果存在）
#         if np.isnan(keypoints[region]).all():
#             continue
#         # print(keypoints[region])
#         # 绘制该区域的5个关键点
#         for kp in range(kps_per_region):
#             x, y = keypoints[region, kp]
#
#             conf = confidences[region]
#
#             # 仅显示置信度>0.5的点
#             if conf > 0.5:
#                 plt.scatter(x, y,
#                             color=colors(region),
#                             s=50,
#                             marker=f'${region}_{kp}$'  # 用编号标记区域和关键点
#                             )
#
#     plt.axis('off')
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     dataset = MultiRegionDataset(
#         root_dir="../data/spine/images",
#         annotation_csv="../data/annotations/train.csv",
#         transform=KeypointTransform(split='train')
#     )
#
#     dataloader = DataLoader(
#         dataset,
#         batch_size=4,
#         shuffle=True,
#         collate_fn=lambda batch: {
#             'image': torch.stack([x['image'] for x in batch]),
#             'keypoints': torch.stack([x['keypoints'] for x in batch]),
#             'confidences': torch.stack([x['confidences'] for x in batch]),
#             'age': torch.stack([x['age'] for x in batch]),
#             'gender': torch.stack([x['gender'] for x in batch])
#         }
#     )
#
#     sample = next(iter(dataloader))
#     print(f"图像尺寸: {sample['image'].shape}")  # [4, 3, 512, 512]
#     print(f"关键点尺寸: {sample['keypoints'].shape}")  # [4, 17, 5, 2]
#     print(f"置信度尺寸: {sample['confidences'].shape}")  # [4, 17]
#     print(sample['image'])
#
#     # 绘制第一个样本
#     plot_sample(sample, idx=0)

# ---
# import matplotlib.pyplot as plt
# import numpy as np
#
# data = """257.6441,18.263988,230.80519,2.8931847,292.22766,11.951765,223.58133,23.444767,283.22943,35.56311,0.14086929
# 245.85294,61.51371,222.8658,39.325645,280.95947,51.69395,211.32376,69.68152,269.2883,85.888374,0.74934524
# 226.14816,105.66148,205.60706,82.54718,260.62115,97.75603,192.82191,112.10493,246.73593,129.42422,0.7754217
# 202.98909,146.5367,187.19566,121.38743,234.42859,140.87178,170.72763,151.62886,219.24579,172.1393,0.5859316
# 178.34412,190.0745,159.97093,164.01918,211.51303,181.02199,143.89984,199.38042,198.19453,215.44223,0.79013807
# 158.94614,234.8608,137.3106,212.11598,192.37698,225.44794,124.28011,246.83472,182.35127,255.62341,0.8034638
# 146.13333,282.59445,117.664085,261.2618,178.20494,266.17526,112.914825,302.74777,176.9576,299.72174,0.73594284
# 146.69756,334.2186,110.62996,319.53296,176.7485,312.93628,115.507904,358.8802,184.05484,346.43738,0.7634096
# 161.70805,389.4906,123.96622,377.96762,186.1713,363.90533,136.78857,416.89157,198.85313,398.95276,0.70409423
# 182.06404,445.82404,140.575,436.03568,204.40776,413.76233,158.36293,477.68802,224.2667,455.1536,0.8421674
# 210.06859,505.09015,168.06522,495.8341,233.64267,474.4359,185.17737,536.2027,251.85641,514.0713,0.77792305
# 234.36513,569.4273,192.64407,556.0693,261.40994,538.9498,205.80617,600.2851,276.44055,582.9296,0.7961093
# 249.97557,637.67456,209.46053,620.183,281.63873,609.8681,217.74551,665.46844,291.07913,655.4271,0.78752196
# 258.32336,710.51416,220.00766,690.1902,293.5035,686.478,221.9225,734.6675,298.00745,730.8681,0.7705383
# 257.81232,785.80206,217.12073,758.455,304.29218,766.5941,209.4158,805.1062,299.4413,812.75195,0.7601752
# 243.01587,862.6554,198.71367,831.5069,297.61664,841.93604,186.19736,882.8045,290.00165,894.0625,0.8364565
# 229.42041,937.431,182.08896,913.23566,281.247,919.7362,175.83371,957.7705,277.31647,959.9382,0.48529908"""
#
# # 图像尺寸
# image_width = 512
# image_height = 1024
#
# # 创建画布（尺寸与图像分辨率一致）
# fig = plt.figure(figsize=(5.12, 10.24), dpi=100)  # 1024px/100dpi=10.24in
# ax = fig.add_subplot(111)
#
#
# # 关键设置：透明背景 & 隐藏坐标元素
# fig.patch.set_alpha(0.0)  # 画布透明
# ax.axis('off')             # 隐藏坐标轴
# ax.set_facecolor('none')   # 绘图区透明
#
# # 设置图像坐标系
# ax.set_xlim(0, 512)
# ax.set_ylim(1024, 0)  # Y轴反向
# ax.set_aspect('equal')
#
# # 生成颜色（每行一个颜色）
# colors = plt.cm.tab20(np.linspace(0, 1, len(data.split('\n'))))
#
# for row_idx, line in enumerate(data.strip().split('\n')):
#     parts = list(map(float, line.split(',')))
#     x_coords = parts[0::2][:5]  # 提取x1~x5
#     y_coords = parts[1::2][:5]  # 提取y1~y5
#
#     # 绘制当前组的5个点
#     ax.scatter(x_coords, y_coords,
#                color=colors[row_idx],
#                s=50,
#                label=f'Group {row_idx + 1}',
#                edgecolors='white',
#                linewidth=0.5)
#
# # 保存为透明PNG（重要参数）
# plt.savefig('annotations.png',
#            transparent=True,  # 强制透明
#            bbox_inches='tight',  # 去除白边
#            pad_inches=0,
#            dpi=100)
#
# # 关闭画布释放内存
# plt.close(fig)