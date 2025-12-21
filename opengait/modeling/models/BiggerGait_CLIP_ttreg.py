import os
import sys
import torch
import torch.nn as nn
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from functools import partial

# 路径设置：请根据实际情况修改
TTR_REPO_PATH = '/home/browei/repo/test-time-registers'
if TTR_REPO_PATH not in sys.path:
    sys.path.insert(0, TTR_REPO_PATH)
clip_sub_path = os.path.join(TTR_REPO_PATH, 'clip')
if clip_sub_path not in sys.path:
    sys.path.insert(0, clip_sub_path)

try:
    from clip_state import load_clip_state
    from shared.hook_manager import HookMode
except ImportError:
    print("[Warning] TTR modules not found. Ensure paths are correct.")

from .BigGait_utils.BigGait_GaitBase import *

# 辅助类
class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax, Relu, Up=True):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim//2),
                nn.BatchNorm1d(source_dim//2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim//2, target_dim))
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim//2),
                    nn.BatchNorm1d(source_dim//2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim//2, source_dim))
        else:
            self.down_sampling = nn.Linear(source_dim, target_dim)
            if Up:
                self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()
        self.Up = Up

    def forward(self, x):
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
            if self.Up:
                u_x = self.up_sampling(d_x)
                return d_x, torch.mean(self.mse(u_x, x))
            else:
                return d_x, None
        else:
            if self.Up:
                u_x = self.up_sampling(d_x)
                return torch.sigmoid(self.bn_t(d_x)), torch.mean(self.mse(u_x, x))
            else:
                return torch.sigmoid(self.bn_t(d_x)), None

class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

# 主模型
class BiggerGait__CLIP_ttreg(BaseModel):
    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg['source_dim']
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.gradient_checkpointing = model_cfg.get("gradient_checkpointing", False)

        # 使用支持 temb 的 ShareTime_2B
        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)
        self.Pre_Conv = nn.Sequential(nn.Identity())

        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.f4_dim*4, affine=False),
                nn.Conv2d(self.f4_dim*4, self.f4_dim//2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size*2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])
        
        # 确保 Mask_Branch 接收 Relu 参数
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        
        self.t_channel = self.f4_dim
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

        # TTR Config
        self.num_registers = model_cfg.get("num_registers", 5) 
        self.ttr_scale = model_cfg.get("ttr_scale", 1.0)
        self.ttr_normal_values = model_cfg.get("ttr_normal_values", "zero")
        self.register_neurons_path = model_cfg.get("register_neurons_path", "")
        self.ttr_config_path = model_cfg.get("ttr_config_path", "")

    def init_CLIP(self):
        import yaml
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载 TTR 配置
        ttr_config = {}
        if os.path.exists(self.ttr_config_path):
            with open(self.ttr_config_path, 'r') as f:
                ttr_config = yaml.safe_load(f)
        else:
            # 默认配置
            ttr_config = {
                "backbone_name": "openai/clip-vit-large-patch14", 
                "detect_outliers_layer": -2, 
                "register_norm_threshold": 5
            }
        
        if self.pretrained_lvm and os.path.exists(self.pretrained_lvm):
            ttr_config["backbone_name"] = self.pretrained_lvm
        ttr_config["device"] = device

        self.msg_mgr.log_info(f"[TTR] Init CLIP with: {ttr_config}")
        
        # 加载模型
        state = load_clip_state(ttr_config)
        self.Backbone = state["model"]
        self.hook_manager = state["hook_manager"]
        self.Backbone.config.output_hidden_states = True

        if self.training and self.gradient_checkpointing:
            self.Backbone.gradient_checkpointing_enable()

        # 注册 TTR Hooks
        self.hook_manager.reinit(mode=HookMode.INTERVENE)
        neurons_to_ablate = {}
        if os.path.exists(self.register_neurons_path):
            loaded_data = torch.load(self.register_neurons_path, map_location="cpu")
            for item in loaded_data:
                layer = int(item[0])
                neuron = int(item[1])
                if layer not in neurons_to_ablate: neurons_to_ablate[layer] = []
                neurons_to_ablate[layer].append(neuron)
            self.msg_mgr.log_info(f"[TTR] Loaded registers for {len(neurons_to_ablate)} layers.")
        
        self.hook_manager.intervene_register_neurons(
            num_registers=self.num_registers,
            neurons_to_ablate=neurons_to_ablate,
            scale=self.ttr_scale,
            normal_values=self.ttr_normal_values
        )
        self.hook_manager.finalize()
        
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)

    def init_Mask_Branch(self):
        load_dict = torch.load(self.pretrained_mask_branch, map_location="cpu")['model']
        self.Mask_Branch.load_state_dict(load_dict, strict=True)
        self.msg_mgr.log_info(f"=> MaskBranch loaded: {self.pretrained_mask_branch}")

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        
        self.init_CLIP()
        self.init_Mask_Branch()
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)
        self.msg_mgr.log_info("=> Init successfully")

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def forward(self, inputs):
        self.Mask_Branch.eval()
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//10)+1, dim=1)
        all_outs = []
        
        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                n,s,c,h,w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w')
                outs = self.preprocess(rgb_img, self.image_size, self.image_size)
                
                # Forward Pass (Hooks active)
                outputs = self.Backbone(pixel_values=outs)
                hidden_states = outputs.hidden_states 

                # [TTR 修正逻辑]
                # TTR 会在序列末尾添加 Register Tokens。
                # 原始 CLIP ViT 输出: [B, 1+HW, C] (1是CLS)
                # TTR CLIP ViT 输出:  [B, 1+HW+Regs, C]
                # 我们需要只保留 CLS(索引0) 和 Patches(索引1到1+HW)
                
                # 计算 Patch 数量 (H/14 * W/14)
                num_patches = (self.image_size // 14) * (self.image_size // 14)
                
                # 最后一层的 Patch Features (去掉 CLS 和 Registers)
                # 索引 1 到 1+num_patches
                outs_last1 = hidden_states[-1][:, 1 : 1 + num_patches].contiguous()
                outs_last1 = rearrange(outs_last1.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) (h w) c')
                outs_last1 = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs_last1)
                
                # Multi-scale Feature Fusion
                # 同样对每一层 hidden_state 进行切片，只保留 Image Patches
                valid_layers = [h[:, 1 : 1 + num_patches] for h in hidden_states[1:]]
                appearance = torch.concat(valid_layers, dim=-1).contiguous()
                
                appearance = rearrange(appearance.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) c h w')
                appearance = self.Pre_Conv(appearance)
                appearance = rearrange(appearance, 'n c h w -> n (h w) c')
                
                # CLIP-Large 24层
                appearance = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*24, elementwise_affine=False)(appearance)
                appearance = rearrange(appearance.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) c h w')
                del hidden_states
            
                human_mask_ori, _ = self.Mask_Branch(outs_last1.view(-1, self.f4_dim))
                human_mask_ori = (human_mask_ori[:,1] > 0.5).float().view(n*s, 1, self.image_size//14, self.image_size//14)
                human_mask_ori = self.preprocess(human_mask_ori, self.sils_size*2, self.sils_size).detach()
                del outs_last1

            human_feat = list(torch.chunk(appearance, 24, dim=1))
            human_feat = [torch.cat(human_feat[i:i+4], dim=1) for i in range(0, 24, 4)]
            for i in range(self.num_FPN):
                human_feat[i] = self.HumanSpace_Conv[i](human_feat[i])
            human_feat = torch.concat(human_feat, dim=1)

            human_feat = human_feat * (human_mask_ori > 0.5).float()
            human_feat = rearrange(human_feat.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> n c s h w')

            outs = self.Gait_Net.test_1(human_feat)
            all_outs.append(outs)

        embed_list, log_list = self.Gait_Net.test_2(torch.cat(all_outs, dim=2), seqL)
        
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
                    'image/human_mask': self.min_max_norm(human_mask_ori.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()).clamp(0,1),
                },
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        else:
            retval = {
                'training_feat': {}, 'visual_summary': {},
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        return retval