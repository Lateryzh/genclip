import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid


import pandas as pd

# import hfai
# import hfai.distributed as dist
# from hfai.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from models.vqgan import VQGAN3
from models.clip import clip_vit_b32
from models.gpt import __dict__ as gpts
# from datasets.gpt2 import __dict__ as datasets
from datasets.gpt3 import MultiRegionDataset, KeypointTransform,PairedDataset
from datasets.statistic import mean, std
from utils import *

from itertools import chain  # 在文件顶部导入

'''
version:
    2去掉分布式
    3改成有监督的训练,没跑通，把监督和关键点放在vqgan里面了
    4改成有监督的训练，把监督放在train里面，把关键点放到gpt里面去映射。感觉会更简单一些，也更符合原本的这个clip-gen的设计
    
    
    data/
├── spine/
│   ├── images/                  # 原始图像目录（按患者ID分类）
│   │   ├── patient001/          # 单个患者影像（多张配对图像）
│   │   │   ├── view1.png        # 同一患者不同视角/时间点的图像
│   │   │   └── view2.png
│   │   ├── patient002/
│   │   └── ...
│   │
│   └── annotations/
│       ├── train.csv           # 训练集标注（包含配对关系）
│       ├── val.csv            # 验证集标注
│       └── test.csv           # 测试集标注（可选）

数据集类型	最小患者数	最小图像数	配对要求
训练集	≥3	≥6	每个患者≥2张配对图像
验证集	≥1	≥2	完整配对链 (A→B→C→A)
测试集	≥2	≥4	包含未见过的配对组合

# train.csv 片段
patient_id,image_path,pair_image_path,keypoints_x,keypoints_y,confidence
001,patient001/view1.png,patient001/view2.png,"120,150,...,300","80,85,...,440",0.95
002,patient002/view1.png,patient002/view3.png,"50,70,...,200","60,65,...,380",0.88

'''

###########################################
# CONFIG
###########################################

parser = argparse.ArgumentParser(description="Train GPT")
parser.add_argument("--ds", type=str, default="spine", help="dataset name")
parser.add_argument("--gpt", type=str, default="gpt2_medium", help="GPT model")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--bs", type=int, default=2, help="batch size")

#新增代码，新增内容为：
parser.add_argument("--kp_loss_weight", type=float, default=0.5, help="keypoint loss weight")

parser.add_argument("--vqgan_ckpt", type=str, default='./pretrained/vqgan_coco.pt',help="VQGAN pretrained model")
args = parser.parse_args()

gpt_name = args.gpt
dataset_name = args.ds
batch_size = args.bs
dropout = args.dropout
vqgan_ckpt = args.vqgan_ckpt

codebook_size = 16384
embed_dim = 256

#新增代码，新增内容为：
keypoint_dim = 17 * 5 * 2  # 17区域×5关键点×2坐标


normalize_clip = True
loss_clip_weight = 0

use_amp = False
enabled_warmup = True

epochs = 200
warmup_epochs = 20
min_lr = 0
base_lr = 5e-4 / 256

save_path = Path(f"output/gpt/{dataset_name}")

# sample
top_k = 500
top_p = 0.95

writer = None


def log_recon_images(name, x, x_recon, step):
    std1 = torch.tensor(std).view(1, -1, 1, 1).to(x)
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).to(x)
    img = torch.cat([x_recon, x], dim=0)  # [2 * N, 3, H, W]
    img = img * std1 + mean1
    img = make_grid(img, nrow=x.size(0))
    writer.add_image(name, img.clamp(0, 1), step)


def logits_to_z(vqgan, logits):
    '''
    logits_to_z：将GPT的输出转换为VQGAN的潜在空间表示
     probs = F.softmax(logits, dim=-1)
    # 使用可导的argmax近似（Straight-Through Estimator）
    onehot = probs + (onehot - probs).detach()  
    # 查表得到潜在表示
    z = onehot @ vqgan.quantizer.embedding.weight  
    return z  # 形状 [B, L, E]

    '''
    # logits, probs: size (B, L, vocab_size)
    probs = F.softmax(logits, dim=-1)
    embedding = vqgan.quantizer.embedding.weight  # (vocab_size, E)
    vocab_size, E = embedding.shape

    # argmax:  size (B, L)
    # one-hot: size (B, L, vocab_size)
    argmax = torch.argmax(logits, dim=2)
    onehot = F.one_hot(argmax, num_classes=vocab_size).float()

    # quantize
    onehot = probs + (onehot - probs).detach()

    B, L, vocab_size = onehot.shape
    z = onehot.view(B * L, vocab_size) @ embedding
    z = z.view(B, L, E)

    return z

#修改前
# def train(dataloader, gpt, vqgan, clip, optimizer, scheduler, scaler, epoch, start_step, best_score):
#修改后，改动的内容为:
def train(dataloader, gpt, vqgan, clip, optimizer, scheduler, scaler, epoch, start_step, best_score):

    gpt.train()
    steps_per_epoch = len(dataloader) + start_step
    # print('enumerate(dataloader)', enumerate(dataloader))
    for step, batch in enumerate(dataloader):

        step += start_step
        lr = scheduler.step(epoch + step / steps_per_epoch)
        # x, clip_x = [t.cuda() for t in batch[:2]]  # images
        # 修改后：
        # 获取成对数据


        #修改前
        # clip_x = batch['clip_imgs'].cuda()
        #修改后，改动的内容为:
        clip_x, keypoints = batch['clip_imgs'].cuda(), batch['keypoints'].cuda()
        print("keypoints: ", keypoints.shape)
        print("clip_x: ", clip_x.shape)

        # #修改后，改动的内容为: 正确应使用成对输入
        # 获取成对数据（已合并为[B*2, ...]）
        vqgan_imgs = batch['vqgan_imgs'].cuda()  # [B*2, C, H, W]
        # print("vqgan_imgs: ", vqgan_imgs)
        print("vqgan_imgs.shape: ", vqgan_imgs.shape)

        # prepare data
        with torch.no_grad():

            #修改前
            # _, _, indices = vqgan.encode(x)

            # 传入模型时保持合并状态
            outputs = vqgan(vqgan_imgs, keypoints=keypoints)

            indices = indices.view(indices.shape[0], -1).detach()  # [B, h * w]  # VQGAN生成的indices是一个形状为[B, 16, 16]的序列，被展平为[B, 256]的序列输入到GPT中。在GPT的输入阶段，这个序列会被处理，因此位置编码应该在这里添加。

            '''
            可能的步骤包括：

            修改GPT模型的输入部分，使其能够接收位置编码。
            在输入到GPT之前，生成位置编码并与现有的输入结合。
            调整模型的forward方法以处理新的输入结构。
            
            '''
            embeddings = clip.encode_image(clip_x)  # [B, 512]

        L = indices.size(1)

        #新增代码，新增内容为：
        B = x.size(0)

        keypoints = keypoints.view(B, -1)  # 展平关键点 [B, 17 * 5 * 2]

        input_tokens = indices[:, :(L - 1)].contiguous()  # [B, L]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = gpt(input_tokens, embeddings)

        loss_gpt = F.cross_entropy(logits.view(-1, logits.size(-1)), indices.view(-1))

        # CLIP embedding reconstruction loss
        if loss_clip_weight > 0:
            z = logits_to_z(vqgan, logits).view(-1, 16, 16, 256)
            z = z.permute(0, 3, 1, 2)
            x_recon = vqgan.decode(z)  # [B, 3, H, W]
            clip_x_recon = F.interpolate(x_recon, size=224, mode='bilinear')
            clip_embeds_recon = clip.encode_image(clip_x_recon)  # [B, 512]
            loss_clip = F.mse_loss(embeddings, clip_embeds_recon)
        else:
            loss_clip = torch.tensor(0).to(x)



        #修改前
        loss = loss_gpt + loss_clip_weight * loss_clip
        #修改后，改动的内容为:
        loss = loss_gpt + loss_clip_weight * loss_clip + outputs['total_loss'] + outputs['quant_loss']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 修改前 (包含分布式和hfai框架逻辑)
        # rank = dist.get_rank()
        # if rank == 0 and hfai.receive_suspend_command():
        #     state = {
        #         "model": gpt.module.state_dict(),  # 注意这里的.module
        #         ...
        #     }
        #     hfai.go_suspend()

        # 修改后 (单机版)
        state = {
            "model": gpt.state_dict(),  # 移除了.module
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step + 1,
            "val_loss": best_score,
        }
        save_model(state, save_path / "latest.pt")

        # log
        # world_size = dist.get_world_size()
        for t in [loss, loss_gpt, loss_clip]:
            # dist.all_reduce(t)
            # t.div_(world_size)
            t.mean()

        total_steps = epoch * steps_per_epoch + step
        if step % 10 == 0:
            mem_used = torch.cuda.max_memory_reserved() // (1 << 20)
            print(f"Epoch: {epoch}, Step: {step}, loss: {loss:.3f}, loss_gpt: {loss_gpt:.3f}, loss_clip: {loss_clip:.3f}, "
                  f"lr: {lr:.5f}, MemUsed: {mem_used} MiB")

            # if rank == 0:
            writer.add_scalar("train/loss", loss, total_steps)
            writer.add_scalar("train/loss_gpt", loss_gpt, total_steps)
            writer.add_scalar("train/loss_clip", loss_clip, total_steps)
            writer.add_scalar("train/lr", lr, total_steps)

        if total_steps % 200 == 0:
            # sample image
            z_idx = gpt.module.sample(embeddings, steps=16 * 16, top_k=top_k, top_p=top_p)  # [B, 16*16] # 自回归生成索引
            with torch.no_grad():
                z_idx = z_idx.view(-1, 16, 16)
                z = vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
                z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
                x_recon = vqgan.decode(z)  # [B, 3, 256, 256]　# 解码为图像

            log_recon_images("train/from-images", x, x_recon, total_steps)


@torch.no_grad()
def validate(dataloader, gpt, vqgan, clip, epoch):

    gpt.eval()
    print("dataloader: ",dataloader)
    # 添加空数据集检查
    if len(dataloader) == 0:
        raise RuntimeError("Validation dataloader is empty. Check dataset configuration")

    total = torch.tensor(0.0).cuda()
    val_loss = torch.tensor(0.0).cuda()

    for batch_idx, batch in enumerate(dataloader):
        try:
            # 添加批次维度检查
            if batch[0].shape[0] == 0:
                print(f"Warning: Empty batch {batch_idx} in validation")
                continue

            # x, clip_x = [t.cuda() for t in batch[:2]]  # 显式获取前两个元素
            # 正确应使用字典键访问
            clip_x = batch['clip_imgs'].cuda()
            keypoints = batch['keypoints'].cuda()

            batch_size = x.size(0)

            # 添加编码有效性检查
            with torch.no_grad():
                _, _, indices = vqgan.encode(x)
                if indices.nelement() == 0:
                    print(f"Invalid indices in batch {batch_idx}")
                    continue

                indices = indices.view(batch_size, -1)
                embeddings = clip.encode_image(clip_x)

            # 前向计算
            L = indices.size(1)
            if L <= 1:
                print(f"Invalid sequence length {L} in batch {batch_idx}")
                continue

            logits = gpt.module(indices[:, :(L - 1)], embeddings)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), indices.view(-1))

            # 累加时使用确定值
            val_loss += loss.detach() * batch_size
            total += batch_size

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    # 添加最终有效性检查
    if total == 0:
        print("Validation failed: No valid batches processed")
        return float('inf')

    avg_loss = val_loss.item() / total.item()
    print(f"Validation completed: {total.item()} samples | Loss: {avg_loss:.4f}")

    # if dist.get_rank() == 0:
    # decode the last batch
    z_idx = gpt.module.sample(embeddings, steps=256, top_k=top_k, top_p=top_p)  # [-1, 16*16]
    z_idx = z_idx.view(-1, 16, 16)
    z = vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
    z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
    x_recon = vqgan.decode(z)  # [B, 3, H, W]

    writer.add_scalar("val/val_loss", val_loss, epoch)
    log_recon_images("val/from-texts", x, x_recon, epoch)

    # dist.barrier()
    return val_loss


def custom_collate(batch):
    # 解包配对样本
    paired_batch = {
        'vqgan_imgs': [],  # 每个元素是(base_img, pair_img)元组
        'clip_imgs': [],
        'keypoints': [],
        'confidences': [],
        'ages': [],
        'genders': []
    }
    vqgan_imgs = torch.cat([item['vqgan_imgs'] for item in batch], dim=0)

    # 重组数据结构
    for sample in batch:
        # 处理图像对
        vqgan_base, vqgan_pair = sample['vqgan_imgs']
        clip_base, clip_pair = sample['clip_imgs']
        kp_base, kp_pair = sample['keypoints']
        conf_base, conf_pair = sample['confidences']
        # 将配对样本视为独立样本
        paired_batch['vqgan_imgs'].extend([vqgan_base, vqgan_pair])
        paired_batch['clip_imgs'].extend([clip_base, clip_pair])
        paired_batch['keypoints'].extend([kp_base, kp_pair])
        paired_batch['confidences'].extend([conf_base, conf_pair])
        paired_batch['ages'].extend([sample['ages'], sample['ages']])  # 元数据复制
        paired_batch['genders'].extend([sample['genders'], sample['genders']])

    # 转换为张量
    return {
        'vqgan_imgs': vqgan_imgs,
        'clip_imgs': torch.stack(paired_batch['clip_imgs']),
        'keypoints': torch.stack(paired_batch['keypoints']),
        'confidences': torch.stack(paired_batch['confidences']),
        'ages': torch.stack(paired_batch['ages']),
        'genders': torch.stack(paired_batch['genders'])
    }

def main():
    log_path = save_path / "runs"
    save_path.mkdir(exist_ok=True, parents=True)
    # rank, world_size = init_dist(local_rank)
    torch.cuda.set_device(0)
    backup(__file__, save_path)


    # 固定随机种子
    torch.manual_seed(42)


    global writer
    writer = SummaryWriter(log_path)

    # total_batch_size = batch_size * world_size
    total_batch_size = batch_size
    lr = base_lr * total_batch_size

    ##################################
    # VQGAN
    ##################################
    #修改前
    # vqgan = VQGAN(codebook_size, embed_dim).cuda().eval().requires_grad_(False)  # 冻结参数的图像编解码器

    #修改后，改动的内容为:
    vqgan = VQGAN3(codebook_size, embed_dim, keypoint_dim=keypoint_dim).cuda().eval().requires_grad_(False)  # 冻结参数的图像编解码器

    state = torch.load(vqgan_ckpt, map_location='cpu',weights_only=True)

    #修改后，改动的内容为: 加入strict=False
    vqgan.load_state_dict(state['model'], strict=False)
    print(f"Loaded VQGAN model from {vqgan_ckpt}, epoch {state['epoch']}")

    # print("成功加载的权重:")
    # for key in state['model']:
    #     if key in vqgan.state_dict():
    #         print(f"✓ {key}")
    #     else:
    #         print(f"✗ {key} (冗余参数)")
    #
    # print("\n新增未初始化的参数:")
    # for key in vqgan.state_dict():
    #     if key not in state['model']:
    #         print(f"☆ {key}")

    # ！！！在这里添加冻结逻辑！！！
    # ----------------------------------
    # 冻结所有参数（除关键点解码器）
    for param in vqgan.parameters():
        param.requires_grad = False  # 关闭梯度计算

    # 解冻关键点解码器
    for param in vqgan.kp_decoder.parameters():
        param.requires_grad = True  # 开启梯度计算
    # ----------------------------------

    ##################################
    # GPT
    ##################################
    gpt = gpts[gpt_name](vocab_size=codebook_size, dropout=dropout)  # 可训练的GPT模型
    # gpt = DistributedDataParallel(gpt, device_ids=[local_rank])train_gpt.py
    # gpt = gpt.cuda()
    # 原代码（可能使用了DataParallel包装）：
    # gpt = torch.nn.DataParallel(gpt).cuda()
    # 改为普通模式（如果使用单卡）：
    gpt = gpt.cuda()

    ##################################
    # CLIP
    ##################################
    clip = clip_vit_b32(pretrained=True).cuda().eval().requires_grad_(False)  # 冻结的CLIP模型
    clip = CLIPWrapper(clip, normalize=normalize_clip)

    ##################################
    # datasets
    ##################################
    train_dataset = PairedDataset(
        root_dir="data/spine/images",
        annotation_csv="data/annotations/train.csv",
        transform=KeypointTransform(split='train')
    )

    val_dataset = PairedDataset(
        root_dir="data/spine/images",
        annotation_csv="data/annotations/val.csv",
        transform=KeypointTransform(split='val')
    )




    # 创建DataLoader（无需特殊collate_fn）
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=custom_collate,  # 关键修改
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        collate_fn=custom_collate,  # 关键修改
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # 测试数据集输出
    # test_dataset = PairedDataset(...)
    # sample = test_dataset[0]
    # print("Dataset keys:", sample.keys())  # 应包含'confidences'
    # print("confidences type:", type(sample['confidences']))  # 应为tuple
    # print("confidences shape:", [c.shape for c in sample['confidences']])  # 确认维度

    ##################################
    # scaler & optimizer
    ##################################
    #修改前
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    #修改后，改动的内容为:
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # #修改前
    # optimizer = configure_optimizer(gpt, lr)
    #修改后，改动的内容为:
    # 正确应包含所有可训练参数
    optimizer = configure_optimizer3(
        [gpt, vqgan.kp_decoder],
        lr
    )

    scheduler = CosineLRWarmUp(optimizer, warmup_epochs, epochs, lr, min_lr, enabled=enabled_warmup)

    # load
    best_score = torch.inf
    start_epoch, start_step = 0, 0
    latest_path = save_path / "latest.pt"
    print("latest_path: ",latest_path)
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu",weights_only=True)

        # gpt.module.load_state_dict(ckpt["model"])
        # 修改后（修复键名前缀）：
        state_dict = ckpt["model"]
        # 去除所有参数的'module.'前缀
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # 严格加载模型
        gpt.load_state_dict(state_dict, strict=True)

        # 加载优化器状态（处理可能的参数组不匹配）
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print("成功加载优化器状态。")
            except ValueError as e:
                print(f"警告：优化器状态不兼容，将重新初始化优化器。错误信息：{e}")
                # 重新配置优化器（确保参数正确）
                optimizer = configure_optimizer3([gpt, vqgan.kp_decoder], lr)
        else:
            print("检查点中无优化器状态，跳过加载。")

        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]
        best_score = ckpt["val_loss"]
        print(f"loaded GPT model from epoch {start_epoch}, step {start_step}")
    else:
        print(f"{latest_path} not found, start training from scractch")

    # validate(val_loader, gpt, vqgan, clip, start_epoch - 1)

    # train, validate
    for epoch in range(start_epoch, epochs):
        # resume from epoch and step
        # train_sampler.set_epoch(epoch)
        # train_loader.set_step(start_step)

        '''
        移除 set_epoch 调用
        原代码 train_sampler.set_epoch(epoch) 用于分布式采样器同步 epoch 状态，在非分布式训练中已无意义，直接删除。
        移除 set_step 调用
        train_loader.set_step(start_step) 是原框架（如 hfai）的自定义方法，PyTorch 原生 DataLoader 无此方法，直接删除。
        '''

        train(train_loader, gpt, vqgan, clip, optimizer, scheduler, scaler, epoch, start_step, best_score)

        val_loss = torch.inf
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_loss = validate(val_loader, gpt, vqgan, clip, epoch)

        start_step = 0  # reset
        # save
        # if rank == 0:
        state = {
            # "model": gpt.module.state_dict(),
            "model": gpt.state_dict(),  # 不再需要.module
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "val_loss": min(best_score, val_loss),
            "step": 0,
        }
        save_model(state, latest_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            save_model(state, save_path / f"{epoch:04d}.pt")

        if val_loss < best_score:
            best_score = val_loss
            save_model(state, save_path / "best.pt")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)

