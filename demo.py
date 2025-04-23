import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision.utils import make_grid

from models.vqgan import VQGAN
from models.clip import clip_vit_b32
from models.gpt import __dict__ as models

from datasets.statistic import mean, std
from tokenizer import tokenize
from utils import CLIPWrapper
from PIL import Image

import math
from torchvision.transforms import functional as TF



# DEFAULT_TEXT = 'A photo of a tower in front of a mountain'
DEFAULT_TEXT = 'CHEST Xray'
# 'A photo of a living area with a television and table'
# 'A city bus driving on the city street'
# 'A train being operated on a train track'
# 'The reflection of the house in the water'
# 'A woman is skiing on a white mountain'

parser = argparse.ArgumentParser(description='CLIP-GEN demo')
parser.add_argument('--text', type=str, default=DEFAULT_TEXT, help='input text')
parser.add_argument('--out', type=str, default='chestphoto2.jpg', help='output image path')
parser.add_argument('--cand-size', type=int, default=64, help='number of candidate images')
parser.add_argument('--out-k', type=int, default=8, help='number of sample images to be saved')
args = parser.parse_args()


torch.set_grad_enabled(False)
device = torch.device('cuda', 0)

gpt_name = "gpt2_medium"
dataset_name = "coco"

# codebook_size = 16384
codebook_size = 8192
embed_dim = 256
dropout = 0.1
normalize_clip = True

batch_size = 8
# vqgan_ckpt = f"pretrained/vqgan_{dataset_name}.pt"
vqgan_ckpt = f"/data/yzh/VQGAN-CLIP/taming-transformers/logs/2025-04-05T14-36-19_custom_vqgan_medical/lightning_logs/2025-04-05T14-36-19_custom_vqgan_medical/checkpoints/epoch=32-step=627000.ckpt"
# gpt_ckpt = f"pretrained/gpt_{dataset_name}.pt"
gpt_ckpt = f"output/gpt/spine/latest.pt"

text = args.text
candidate_size = args.cand_size
out_k = args.out_k
top_k = 500
top_p = 0.95
bs = 8  # batch size
assert candidate_size % bs == 0

##################################
# VQGAN
##################################

vqgan = VQGAN(codebook_size, embed_dim).to(device).eval().requires_grad_(False)
state = torch.load(vqgan_ckpt, map_location='cpu')
# vqgan.load_state_dict(state['model'])
vqgan.load_state_dict(state['state_dict'], strict=False)
print(f"Loaded VQGAN model from {vqgan_ckpt}, epoch {state['epoch']}")

##################################
# GPT
##################################
gpt = models[gpt_name](vocab_size=codebook_size, dropout=dropout).to(device).eval()
state = torch.load(gpt_ckpt, map_location='cpu')
gpt.load_state_dict(state['model'],strict=False)
print(f"Loaded GPT model from {gpt_ckpt}, epoch {state['epoch']}")

##################################
# CLIP
##################################
clip = clip_vit_b32(pretrained=True).to(device).eval()
clip = CLIPWrapper(clip, normalize=normalize_clip)


##################################
# sample
##################################
print("Input text:", text)
texts = [text]
texts = tokenize(texts).to(device)

x_recons = []

text_embeddings = clip.encode_text(texts) # [1, 512]
# print('text_embeddings',text_embeddings)
embeds = text_embeddings.expand(bs, -1)
# embeds = img_embeddings.expand(bs, -1)
# print('embeds',embeds.shape)




# 加载图片并提取 CLIP 图像嵌入
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        print('pm实例', self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean())
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 3)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


input_image = Image.open("data/11-F-0010805459-a-25-August-2022.png").convert("RGB")
make_cutouts = MakeCutouts(224, 128,10)
input_image = make_cutouts(TF.to_tensor(input_image).unsqueeze(0).to(device))
print(type(input_image), input_image.shape if hasattr(input_image, 'shape') else input_image)
# input_image = "data/11-F-0010805459-a-25-August-2022.png"
img_embeddings = clip.encode_image(input_image)  # [1, 512]
# embeds = img_embeddings.expand(bs, -1)
embeds = img_embeddings.clone()




for i in range(candidate_size // bs):
    z_idx = gpt.sample(embeds, steps=16 * 16, top_k=top_k, top_p=top_p)  # [-1, 16*16]
    z_idx = z_idx.view(-1, 16, 16)
    z = vqgan.quantizer.decode(z_idx)  # (B, H, W, C)
    z = z.permute(0, 3, 1, 2)  # [B, C, H, W]
    x_recon = vqgan.decode(z)  # [B, 3, H, W]
    x_recons.append(x_recon)

# torch.cuda.empty_cache()
x_recon = torch.cat(x_recons, dim=0)


##################################
# filter by CLIP
##################################

clip_x_recon = F.interpolate(x_recon, 224, mode='bilinear')

img_embeddings = []

for i in range(candidate_size // bs):
    embd = clip.encode_image(clip_x_recon[i * bs:(i+1) * bs])  # [B, 512]
    img_embeddings.append(embd)
    torch.cuda.empty_cache()
img_embeddings = torch.cat(img_embeddings, dim=0)

sim = F.cosine_similarity(text_embeddings, img_embeddings)
topk = sim.argsort(descending=True)[:out_k]
# print("CLIP similarity", sim[topk])


##################################
# display image
##################################

x = x_recon[topk]
std = torch.tensor(std).view(1, -1, 1, 1).to(x)
mean = torch.tensor(mean).view(1, -1, 1, 1).to(x)
img = x.clone()  # [2 * N, 3, H, W]
img = img * std + mean
img = make_grid(img, nrow=min(x.size(0), 4))
img = img.permute(1, 2, 0).clamp(0, 1)
plt.imshow(img.cpu())
plt.title(text)
plt.axis('off')
plt.savefig(args.out, bbox_inches='tight')
print('跑完了')
