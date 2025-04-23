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

from models.vqgan import VQGAN
from models.clip import clip_vit_b32
from models.gpt import __dict__ as gpts
# from datasets.gpt2 import __dict__ as datasets
from datasets.gpt2 import MultiRegionDataset, KeypointTransform
from datasets.statistic import mean, std
from utils import *

'''
去掉分布式

├── 配置解析 (argparse)
├── 模型组件
│   ├── VQGAN (图像编解码器)
│   ├── GPT (生成模型)
│   └── CLIP (语义指导)
├── 数据流水线
│   ├── 分布式采样器
│   └── 数据加载
├── 训练循环
│   ├── 混合精度训练
│   ├── 损失计算（交叉熵 + CLIP重建损失）
│   └── 学习率调度
└── 验证与保存


def main(): 
    初始化分布式环境 --> 加载VQGAN --> 加载GPT --> 加载CLIP --> 数据加载 --> 优化器配置 --> 训练循环

理解核心模型交互（重点！）

1. VQGAN（图像压缩模块）
核心功能：将256x256图像编码为16x16的离散编码（类似将图片转换为16x16的乐高积木编号图）
关键代码段：

_, _, indices = vqgan.encode(x) # 得到编码索引 [B, 16, 16]
z = vqgan.quantizer.decode(z_idx) # 从索引重建特征
x_recon = vqgan.decode(z) # 解码为像素图像


2. CLIP（语义对齐模块）
作用：将图像映射到文本语义空间，提供生成指导

embeddings = clip.encode_image(clip_x) # 提取图像语义特征 [B, 512]
clip_embeds_recon = clip.encode_image(clip_x_recon) # 约束生成图像语义


3. GPT（生成模型）
改造点：将文本生成改为视觉编码生成

logits = gpt(input_tokens, embeddings) # 输入当前编码序列+CLIP语义，预测下一个编码
z_idx = gpt.module.sample(embeddings) # 用CLIP语义引导生成完整编码序列

'''

###########################################
# CONFIG
###########################################

parser = argparse.ArgumentParser(description="Train GPT")
parser.add_argument("--ds", type=str, default="spine", help="dataset name")
parser.add_argument("--gpt", type=str, default="gpt2_medium", help="GPT model")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--bs", type=int, default=2, help="batch size")
parser.add_argument("--vqgan_ckpt", type=str, default='/data/yzh/clip-gen/pretrained/vqgan_coco.pt',help="VQGAN pretrained model")
args = parser.parse_args()

gpt_name = args.gpt
dataset_name = args.ds
batch_size = args.bs
dropout = args.dropout
vqgan_ckpt = args.vqgan_ckpt

codebook_size = 16384
embed_dim = 256
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


def train(dataloader, gpt, vqgan, clip, optimizer, scheduler, scaler, epoch, start_step, best_score):
    '''
    for batch in dataloader:
        # 前向流程
        x → VQGAN.encode → indices (图像编码索引)
        clip_x → CLIP → embeddings (语义特征)

        # GPT生成
        logits = GPT(indices[:-1], embeddings)  # 自回归生成

        # 损失计算
        loss_gpt = CE_loss(logits, indices)  # 交叉熵损失
        loss_clip = ||CLIP(x) - CLIP(GPT→VQGAN.decode(x))|| # 重建语义对齐
    '''

    gpt.train()
    steps_per_epoch = len(dataloader) + start_step
    print('enumerate(dataloader)',enumerate(dataloader))
    for step, batch in enumerate(dataloader):

        step += start_step
        lr = scheduler.step(epoch + step / steps_per_epoch)
        # x, clip_x = [t.cuda() for t in batch[:2]]  # images
        # 修改后：
        x = batch['vqgan_imgs'].cuda()
        # print('x',x)
        print('x.shape',x.shape) # 2, 3, 256, 256

        clip_x = batch['clip_imgs'].cuda()



        # prepare data
        with torch.no_grad():
            _, _, indices = vqgan.encode(x)
            indices = indices.view(indices.shape[0], -1).detach()  # [B, h * w]  # VQGAN生成的indices是一个形状为[B, 16, 16]的序列，被展平为[B, 256]的序列输入到GPT中。在GPT的输入阶段，这个序列会被处理，因此位置编码应该在这里添加。
            '''
            可能的步骤包括：

            修改GPT模型的输入部分，使其能够接收位置编码。
            在输入到GPT之前，生成位置编码并与现有的输入结合。
            调整模型的forward方法以处理新的输入结构。
            '''
            embeddings = clip.encode_image(clip_x)  # [B, 512]

        L = indices.size(1)
        print('199 L',L)
        input_tokens = indices[:, :(L - 1)].contiguous()  # [B, L]
        '''
        在代码中，处理输入的地方主要在训练循环中的以下部分：

        在train函数中，input_tokens是VQGAN的indices截取的前L-1个位置，然后传递给GPT的forward方法。
        在模型的forward过程中，可能需要在嵌入层之后加上位置编码。
        
        因此，修改点可能包括：
        
            在GPT模型的初始化阶段，添加位置编码层，例如nn.Embedding，用于生成位置向量。
            在模型的forward方法中，将输入tokens的嵌入与位置编码相加。
        
        因此，修改点可能包括：

            在GPT模型的初始化阶段，添加位置编码层，例如nn.Embedding，用于生成位置向量。
            在模型的forward方法中，将输入tokens的嵌入与位置编码相加。
            例如，在定义GPT模型时，可以添加一个位置编码层，比如：
            
            class GPT(nn.Module):
            def init(self, vocab_size, dropout=0.1, max_seq_length=256):
            super().init()
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(max_seq_length, embed_dim)
        
        '''
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            print("input_tokens.shape: ",input_tokens.shape)
            print("embeddings.shape: ",embeddings.shape)
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

        loss = loss_gpt + loss_clip_weight * loss_clip

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
            # z_idx = gpt.module.sample(embeddings, steps=16 * 16, top_k=top_k, top_p=top_p)  # [B, 16*16] # 自回归生成索引
            print("embeddings.shape2: ",embeddings.shape)
            z_idx = gpt.sample(embeddings, steps=16 * 16, top_k=top_k, top_p=top_p)  # [B, 16*16] # 自回归生成索引
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

            x, clip_x = [t.cuda() for t in batch[:2]]  # 显式获取前两个元素
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
    z_idx = gpt.sample(embeddings, steps=256, top_k=top_k, top_p=top_p)  # [-1, 16*16]
    z_idx = z_idx.view(-1, 16, 16)
    z = vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
    z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
    x_recon = vqgan.decode(z)  # [B, 3, H, W]

    writer.add_scalar("val/val_loss", val_loss, epoch)
    log_recon_images("val/from-texts", x, x_recon, epoch)

    # dist.barrier()
    return val_loss


def main(local_rank):
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
    vqgan = VQGAN(codebook_size, embed_dim).cuda().eval().requires_grad_(False)  # 冻结参数的图像编解码器
    state = torch.load(vqgan_ckpt, map_location='cpu')
    vqgan.load_state_dict(state['model'])
    print(f"Loaded VQGAN model from {vqgan_ckpt}, epoch {state['epoch']}")

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
    train_dataset = MultiRegionDataset(
        root_dir="data/spine/images",
        annotation_csv="data/annotations/train.csv",
        transform=KeypointTransform(split='train')
    )

    val_dataset = MultiRegionDataset(
        root_dir="data/spine/images",
        annotation_csv="data/annotations/val.csv",
        transform=KeypointTransform(split='val')
    )


    ############################################
    # 添加自定义collate函数处理多类型数据
    ############################################
    def custom_collate(batch):
        return {
            'vqgan_imgs': torch.stack([x[0] for x in batch]),
            'clip_imgs': torch.stack([x[1] for x in batch]),
            'keypoints': torch.stack([x[2] for x in batch]),
            'confidences': torch.stack([x[3] for x in batch]),
            'ages': torch.stack([x[4] for x in batch]),
            'genders': torch.stack([x[5] for x in batch])
        }

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


    ##################################
    # scaler & optimizer
    ##################################
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = configure_optimizer2(gpt, lr)
    scheduler = CosineLRWarmUp(optimizer, warmup_epochs, epochs, lr, min_lr, enabled=enabled_warmup)

    # load
    best_score = torch.inf
    start_epoch, start_step = 0, 0
    latest_path = save_path / "latest.pt"
    print("latest_path: ",latest_path)
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu")

        # gpt.module.load_state_dict(ckpt["model"])
        # 修改后（修复键名前缀）：
        state_dict = ckpt["model"]
        # 去除所有参数的'module.'前缀
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # 严格加载模型
        gpt.load_state_dict(state_dict, strict=True)

        optimizer.load_state_dict(ckpt["optimizer"])
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
    main(local_rank=0)
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)

