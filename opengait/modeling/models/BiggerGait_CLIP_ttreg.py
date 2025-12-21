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
        self.chunk_size = model_cfg.get("chunk_size", 16)

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
        """
        使用 TTR OpenCLIP 逻辑加载模型
        """
        import yaml
        
        # 1. 确定设备
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # 2. 加载 TTR 配置文件
        ttr_config = {}
        if os.path.exists(self.ttr_config_path):
            self.msg_mgr.log_info(f"[TTR] Loading config from: {self.ttr_config_path}")
            with open(self.ttr_config_path, 'r') as f:
                ttr_config = yaml.safe_load(f)
        else:
            self.msg_mgr.log_info(f"[Error] TTR config not found. Using fallback.")
            # Fallback for OpenCLIP ViT-Large
            ttr_config = {
                "model_name": "ViT-L-14", 
                "pretrained": "openai",
                "device": device
            }
        
        # 强制覆盖 device
        ttr_config["device"] = device
        
        self.msg_mgr.log_info(f"[TTR] Initializing OpenCLIP with: {ttr_config}")

        # 3. 调用 load_clip_state
        # 返回的是一个 dict: {'model': ..., 'hook_manager': ..., ...}
        state = load_clip_state(ttr_config)
        
        # 4. 获取模型
        full_model = state["model"]
        # BiggerGait 只需要视觉部分
        self.Backbone = full_model.visual
        self.hook_manager = state["hook_manager"]

        # =====================================================================
        # [严重警告] OpenCLIP 模型默认 forward 不返回 hidden_states (中间层特征)
        # BiggerGait 必须需要中间层特征。
        # 较新版本的 OpenCLIP (v2.20+) 的 VisionTransformer 支持 set_grad_checkpointing
        # 和返回中间层。如果不确定 OpenCLIP 版本，这里可能会报错。
        # =====================================================================
        
        # 尝试开启梯度检查点 (如果支持)
        if self.training and self.gradient_checkpointing:
            try:
                self.Backbone.set_grad_checkpointing(True)
                self.msg_mgr.log_info("Gradient Checkpointing Enabled for OpenCLIP!")
            except AttributeError:
                self.msg_mgr.log_info("Gradient Checkpointing NOT supported by this OpenCLIP version.")

        # 5. TTR 干预 (与之前逻辑相同)
        self.hook_manager.reinit(mode=HookMode.INTERVENE)
        neurons_to_ablate = {}
        if os.path.exists(self.register_neurons_path):
            loaded_data = torch.load(self.register_neurons_path, map_location="cpu")
            for item in loaded_data:
                # 兼容不同格式
                if len(item) == 3: layer, neuron, _ = item
                else: layer, neuron = item
                layer, neuron = int(layer), int(neuron)
                if layer not in neurons_to_ablate: neurons_to_ablate[layer] = []
                neurons_to_ablate[layer].append(neuron)
            self.msg_mgr.log_info(f"[TTR] Registers loaded for {len(neurons_to_ablate)} layers.")
        
        self.hook_manager.intervene_register_neurons(
            num_registers=self.num_registers,
            neurons_to_ablate=neurons_to_ablate,
            scale=self.ttr_scale,
            normal_values=self.ttr_normal_values
        )
        self.hook_manager.finalize()
        
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)

        # [关键] 注册 Hooks 来捕获中间层输出
        self.intermediate_features = {}
        self.hook_handles = []
        
        # 定义 Hook 函数
        def get_activation(name):
            def hook(model, input, output):
                # output 可能是 tensor 或 tuple
                # 对于 Transformer Block，output 通常是 [B, L, C]
                if isinstance(output, tuple):
                    output = output[0]
                self.intermediate_features[name] = output
            return hook

        # 遍历模型的 Transformer 层并注册 Hook
        # 注意：不同库的层命名不同
        # HuggingFace: model.vision_model.encoder.layers
        # OpenCLIP:    model.visual.transformer.resblocks
        
        layers_to_hook = []
        if hasattr(self.Backbone, 'vision_model'): # HuggingFace
            layers_to_hook = self.Backbone.vision_model.encoder.layers
        elif hasattr(self.Backbone, 'transformer'): # OpenCLIP (部分版本)
            layers_to_hook = self.Backbone.transformer.resblocks
        
        for i, layer in enumerate(layers_to_hook):
            # 我们Hook每一层的输出
            handle = layer.register_forward_hook(get_activation(i))
            self.hook_handles.append(handle)
            
        self.msg_mgr.log_info(f"Registered hooks for {len(layers_to_hook)} layers.")

        self.msg_mgr.log_info("=> OpenCLIP Init successfully")

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

        CHUNK_SIZE = self.chunk_size 
        
        # 逻辑：将总帧数切分成更小的块
        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // CHUNK_SIZE) + 1, dim=1)
        all_outs = []
        
        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                n,s,c,h,w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
                outs = self.preprocess(rgb_img, self.image_size, self.image_size)
                
                # 清空缓存
                self.intermediate_features = {}
                
                # Forward (Hook 会自动运行并将结果存入 self.intermediate_features)
                # # OpenCLIP 不接受 pixel_values，直接传 tensor
                _ = self.Backbone(outs)
                
                # 收集 Hidden States
                # 假设我们有 24 层，按照索引排序
                # hidden_states = [self.intermediate_features[i] for i in range(24)]
                # 或者更安全地：
                hidden_states = []
                for i in range(len(self.hook_handles)):
                    if i in self.intermediate_features:
                        hidden_states.append(self.intermediate_features[i])
                
                if not hidden_states:
                    raise RuntimeError("Hooks failed to capture any features!")

                # 假设 hidden_states 是 list of [B, N_tokens, C]
                # TTR 会增加 N_tokens 的长度 (Registers)
                
                # 计算标准的 Patch 数量
                num_patches = (self.image_size // 14) * (self.image_size // 14)
                # OpenCLIP ViT 通常第 0 个是 CLS，后面是 Patches，再后面是 Registers (如果有 TTR)
                
                # 取最后一层的 Patch 特征 (排除 CLS 和 Registers)
                # 假设结构: [CLS, Patch_1, ..., Patch_N, Reg_1, ..., Reg_M]
                outs_last1 = hidden_states[-1][:, 1 : 1 + num_patches].contiguous()
                outs_last1 = rearrange(outs_last1.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) (h w) c')
                outs_last1 = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs_last1)
                
                # 多尺度融合
                # 同样切片：只取 Image Patches
                valid_layers = [h[:, 1 : 1 + num_patches] for h in hidden_states] # 全层使用
                # 注意：hidden_states[0] 通常是第一层输出。BiggerGait 似乎从第2层开始取? 
                # 原代码: hidden_states[1:]。这里根据实际情况调整。
                # 假设我们取所有返回的层
                appearance = torch.concat(valid_layers, dim=-1).contiguous()
                
                appearance = rearrange(appearance.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) c h w')
                appearance = self.Pre_Conv(appearance)
                appearance = rearrange(appearance, 'n c h w -> n (h w) c')
                
                # 归一化
                appearance = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim * len(valid_layers), elementwise_affine=False)(appearance)
                appearance = rearrange(appearance.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) c h w')
            
                human_mask_ori, _ = self.Mask_Branch(outs_last1.view(-1, self.f4_dim))
                human_mask_ori = (human_mask_ori[:,1] > 0.5).float().view(n*s, 1, self.image_size//14, self.image_size//14)
                human_mask_ori = self.preprocess(human_mask_ori, self.sils_size*2, self.sils_size).detach()

            # 后续处理逻辑不变
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