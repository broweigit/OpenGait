# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import sys
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from kornia import morphology as morph
import random

# =========================================================================
# 1. 路径设置 & 官方模块导入
# =========================================================================
# 你的 TTR 仓库根目录
TTR_REPO_PATH = '/home/browei/repo/test-time-registers'

# 关键：必须把 dinov2 子目录也加到 sys.path，否则内部 import 会失败
if TTR_REPO_PATH not in sys.path:
    sys.path.insert(0, TTR_REPO_PATH)
dinov2_sub_path = os.path.join(TTR_REPO_PATH, 'dinov2')
if dinov2_sub_path not in sys.path:
    sys.path.insert(0, dinov2_sub_path)

try:
    # 尝试导入官方加载函数
    from dinov2_state import load_dinov2_state
    from shared.hook_manager import HookMode
except ImportError as e:
    print(f"[Error] TTR Import failed: {e}")
    print("请检查 TTR_REPO_PATH 是否指向了正确的 test-time-registers 根目录")

from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import save_image, pca_image
from functools import partial

# ######################################## BiggerGait ###########################################

class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        self.down_sampling = nn.Linear(source_dim, target_dim)
        self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()

    def forward(self, x, mse=True):
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        d_x = F.softmax(d_x, dim=1)
        if mse:
            u_x = self.up_sampling(d_x)
            return d_x, torch.mean(self.mse(u_x, x))
        else:
            return d_x, None

class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

class BiggerGait__DINOv2_ttreg(BaseModel):
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]

        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]

        # set feature dim
        self.f4_dim = model_cfg['source_dim']
        self.num_unknown = model_cfg["num_unknown"]

        # set layer / group / gait_head number
        self.total_layer_num = model_cfg["total_layer_num"] # total layer number is 12
        self.group_layer_num = model_cfg["group_layer_num"] # each group have 2 layers
        self.head_num = model_cfg["head_num"] # 2 gait heads
        assert self.total_layer_num % self.group_layer_num == 0
        assert (self.total_layer_num // self.group_layer_num) % self.head_num == 0
        self.num_FPN = self.total_layer_num // self.group_layer_num

        self.Gait_Net = Baseline_Share(model_cfg)

        self.HumanSpace_Conv = nn.ModuleList([ 
                nn.Sequential(
                    nn.BatchNorm2d(self.f4_dim*self.group_layer_num, affine=False),
                    nn.Conv2d(self.f4_dim*self.group_layer_num, self.f4_dim//2, kernel_size=1),
                    nn.BatchNorm2d(self.f4_dim//2, affine=False),
                    nn.GELU(),
                    nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                    ResizeToHW((self.sils_size*2, self.sils_size)),
                    nn.BatchNorm2d(self.num_unknown, affine=False),
                    nn.Sigmoid()
                ) for _ in range(self.num_FPN)
            ])
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])

        # TTR Configuration
        self.num_registers = model_cfg.get("num_registers", 1) # Default to 1 registers if not specified
        self.ttr_scale = model_cfg.get("ttr_scale", 1.0)
        self.ttr_normal_values = model_cfg.get("ttr_normal_values", "zero")
        self.register_neurons_path = model_cfg.get("register_neurons_path", "./pretrained_LVMs/register_neurons.pt")
        self.ttr_config_path = model_cfg.get("ttr_config_path", "./pretrained_LVMs/dinov2_small_ttr.yaml")

    # def init_DINOv2(self):
    #     # from transformers import Dinov2Config, Dinov2Model
    #     # from transformers.modeling_outputs import BaseModelOutputWithPooling
    #     # config = Dinov2Config.from_pretrained(self.pretrained_lvm + "/config.json")
    #     # self.Backbone = Dinov2Model.from_pretrained(
    #     #     self.pretrained_lvm, 
    #     #     config=config,
    #     # )
    #     # self.Backbone.cpu()
    #     # self.msg_mgr.log_info(f'load model from: {self.pretrained_lvm}')

    #     # Using the logic to avoid namespace conflict if necessary:
    #     try:
    #         # Try standard loading first
    #         from transformers import Dinov2Config, Dinov2Model
    #         config = Dinov2Config.from_pretrained(self.pretrained_lvm + "/config.json")
    #         original_model = Dinov2Model.from_pretrained(
    #             self.pretrained_lvm, 
    #             config=config,
    #         )
    #     except:
    #          # Fallback to torch hub if local loading fails or for standard vit_small
    #          # Or use the manual loading method from previous answer if needed
    #          print("Loading from Torch Hub as fallback...")
    #          original_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    #     # 2. Wrap with HookManager for TTR
    #     # ================= 修改点在这里 =================
    #     # 错误写法 (之前):
    #     # self.hook_manager = HookManager(original_model)
        
    #     # 正确写法: 使用 DINOv2 专用的子类
    #     self.hook_manager = Dinov2HookManager(original_model)
    #     # ==============================================
        
    #     # 3. Initialize TTR (Auto-find register neurons if not provided)
    #     # Note: In a real training loop, you might want to find these once and save them.
    #     # For now, we assume auto-detection or you can pass a fixed dict if known.
    #     self.hook_manager.reinit(mode=HookMode.INTERVENE)
        
    #     # Find or set register neurons. 
    #     # Warning: TTR algorithm typically needs a batch of images to find outliers.
    #     # Since we are in init, we might defer this or use a dummy batch if possible,
    #     # OR we assume the hook manager handles it dynamically/lazily.
    #     # For stability, you might want to manually set `neurons_to_ablate` if you know them for ViT-S.
    #     # If not, HookManager might need a 'warmup' step.
        
    #     # 4. Register Intervention
    #     # We intervene with the specified number of registers.
    #     # Note: We pass empty dict for neurons_to_ablate to let TTR find them or simply use registers.
    #     # If TTR requires finding them first, you'd run algorithm.get_outliers() here.
    #     # --- NEW: Load your custom register neurons ---
    #     reg_path = self.register_neurons_path
        
    #     if os.path.exists(reg_path):
    #         print(f"Loading register neurons from {reg_path}")
    #         register_neurons_list = torch.load(reg_path) # 这是一个 list of (layer, neuron, score)
            
    #         # 转换为 HookManager 需要的格式: {layer: [neuron_idx, ...]}
    #         neurons_to_ablate = {}
    #         for layer, neuron, score in register_neurons_list:
    #             if layer not in neurons_to_ablate:
    #                 neurons_to_ablate[layer] = []
    #             neurons_to_ablate[layer].append(neuron)

    #         # 打印加载的 register neurons 信息
    #         # for layer, neurons in neurons_to_ablate.items():
    #         #     print(f"Layer {layer}: Register Neurons - {neurons}")
    #     else:
    #         print(f"Warning: Register neurons file not found at {reg_path}. Using empty intervention.")
    #         neurons_to_ablate = {}
        
    #     self.hook_manager.intervene_register_neurons(
    #         num_registers=self.num_registers,
    #         neurons_to_ablate=neurons_to_ablate,
    #         scale=self.ttr_scale,
    #         normal_values=self.ttr_normal_values
    #     )
        
    #     # 5. Finalize hooks
    #     self.hook_manager.finalize()
        
    #     self.Backbone = original_model
    #     self.Backbone.cpu()
    #     self.msg_mgr.log_info(f'load model with TTR (regs={self.num_registers}) from: {self.pretrained_lvm}')

    def init_DINOv2(self):
        """
        使用 TTR 官方逻辑加载模型，支持分布式训练设备自动检测和外部配置文件
        """
        import yaml
        
        # 1. 确定当前设备 (适配分布式训练)
        # 在 OpenGait/DDP 中，通常 CUDA_VISIBLE_DEVICES 或 set_device 已经配置好
        # 使用 "cuda" 会自动映射到当前进程分配的 GPU
        if torch.cuda.is_available():
            device = "cuda" 
        else:
            device = "cpu"
            self.msg_mgr.log_info("[Warning] CUDA not available, falling back to CPU.")

        # 2. 加载 TTR 配置文件 (YAML)
        ttr_config = {}
        if os.path.exists(self.ttr_config_path):
            self.msg_mgr.log_info(f"[TTR] Loading config from: {self.ttr_config_path}")
            with open(self.ttr_config_path, 'r') as f:
                ttr_config = yaml.safe_load(f)
        else:
            # 如果找不到文件，使用默认配置或报错
            self.msg_mgr.log_info(f"[Error] TTR config file not found at {self.ttr_config_path}. Using fallback defaults.")
            ttr_config = {
                "backbone_size": "vits14", 
                "detect_outliers_layer": -2, 
                "register_norm_threshold": 5
            }
        
        # [关键] 强制覆盖配置文件中的 device，确保使用当前进程的 GPU
        ttr_config["device"] = device
        
        self.msg_mgr.log_info(f"[TTR] Initializing model with config: {ttr_config}")

        # 3. 调用官方加载函数
        # load_dinov2_state
        state = load_dinov2_state(ttr_config)
        
        self.Backbone = state["model"]
        self.hook_manager = state["hook_manager"]
        
        # 4. 准备干预 (Intervention)
        self.hook_manager.reinit(mode=HookMode.INTERVENE)
        
        # 加载离线计算好的 register neurons
        neurons_to_ablate = {}
        if os.path.exists(self.register_neurons_path):
            self.msg_mgr.log_info(f"[TTR] Loading registers from {self.register_neurons_path}")
            loaded_data = torch.load(self.register_neurons_path, map_location="cpu")
            
            for item in loaded_data:
                if len(item) == 3:
                    layer, neuron, _ = item
                else:
                    layer, neuron = item
                
                layer = int(layer)
                if layer not in neurons_to_ablate:
                    neurons_to_ablate[layer] = []
                neurons_to_ablate[layer].append(int(neuron))
            
            self.msg_mgr.log_info(f"[TTR] Registers loaded for {len(neurons_to_ablate)} layers.")
        else:
            self.msg_mgr.log_info(f"[TTR] Warning: {self.register_neurons_path} not found. No neurons will be ablated.")

        # 5. 注册干预 Hooks
        self.hook_manager.intervene_register_neurons(
            num_registers=self.num_registers,
            neurons_to_ablate=neurons_to_ablate,
            scale=self.ttr_scale,
            normal_values=self.ttr_normal_values
        )
        
        # 6. Finalize
        self.hook_manager.finalize()


    def init_Mask_Branch(self):
        self.msg_mgr.log_info(f'load model from: {self.pretrained_mask_branch}')
        load_dict = torch.load(self.pretrained_mask_branch, map_location=torch.device("cpu"))['model']
        msg = self.Mask_Branch.load_state_dict(load_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.Mask_Branch.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_mask_branch}'")
        self.msg_mgr.log_info('SegmentationBranch Count: {:.5f}M'.format(n_parameters / 1e6))

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('Expect Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        
        self.init_DINOv2()
        self.init_Mask_Branch()
        
        # # Cal GFlops
        if self.training:
            from fvcore.nn import FlopCountAnalysis
            self.eval()
            with torch.no_grad():
                device = torch.distributed.get_rank()
                inputs = ([[torch.randn((1,1,3,448,224),dtype=torch.float32).to(device), torch.rand(1,dtype=torch.float32).to(device)], None, None, None, None],)
                # flops = FlopCountAnalysis(self.to(device), inputs).total()  / 1e9   # GFLOPs 
            self.train()
        
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)
        
        n_parameters = sum(p.numel() for p in self.parameters())
        # if self.training:
        #     self.msg_mgr.log_info('All Backbone Count: {:.5f}M, {:.2f} GFLOPs'.format(n_parameters / 1e6, flops))
        # else:
        #     self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
            
        self.msg_mgr.log_info("=> init successfully")

    # resize image
    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size*2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

# # ############################# For Train ##############################

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # adjust gpu
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//96)+1, dim=1)
        all_outs = []
        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                # get RGB
                n,s,c,h,w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
                outs = self.preprocess(rgb_img, self.image_size)
                
                # huggingface 版本特有output_hidden_states
                # outs = self.Backbone(outs,output_hidden_states=True).hidden_states[1:] # [ns,h*w,c]

                # ========================= 修改开始 =========================
                # 移除所有 Register Token 逻辑，使用 DINOv2 标准 API 获取特征
                # get_intermediate_layers 返回: [(patches, cls_token), (patches, cls_token), ...]
                feat_tuples = self.Backbone.get_intermediate_layers(
                    outs, 
                    n=self.total_layer_num,  # 获取最后 n 层
                    reshape=False,           # 保持 [B, N, C] 形状
                    return_class_token=True  # 同时返回 CLS token
                )
                
                # 将 (patches, cls) 重新组装成 BiggerGait 需要的 [B, Tokens, C]
                # 这里的 Tokens = CLS + Patches
                hidden_states = []
                for patches, cls_token in feat_tuples:
                    # cls_token 形状通常是 [B, C]，需要升维成 [B, 1, C] 才能拼接
                    if cls_token.dim() == 2:
                        cls_token = cls_token.unsqueeze(1)
                    
                    # 拼接: [CLS, Patches]
                    seq = torch.cat([cls_token, patches], dim=1)
                    hidden_states.append(seq)
                
                outs = hidden_states
                # ========================= 修改结束 =========================

                intermediates = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*len(outs), elementwise_affine=False)(torch.concat(outs, dim=-1))[:,1:]
                intermediates = rearrange(intermediates.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
                intermediates = list(torch.chunk(intermediates, self.total_layer_num, dim=1))

                human_mask = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs[-1])[:,1:].contiguous()
                human_mask, _ = self.Mask_Branch(human_mask.view(-1, self.f4_dim), mse=False)
                human_mask = (human_mask[:,1] > 0.5).float() # check which is the foreground at first!!!   0 or 1; 50%;
                human_mask = human_mask.view(n*s, 1, self.image_size//7, self.image_size//14)
                human_mask = self.preprocess(human_mask, self.sils_size).detach().clone()

            intermediates = [torch.cat(intermediates[i:i+self.group_layer_num], dim=1).contiguous() for i in range(0, self.total_layer_num, self.group_layer_num)]
            for i in range(self.num_FPN):
                intermediates[i] = self.HumanSpace_Conv[i](intermediates[i])
            intermediates = torch.concat(intermediates, dim=1)
            intermediates = intermediates * (human_mask > 0.5).to(intermediates)
            intermediates = rearrange(intermediates.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> n c s h w').contiguous()

            outs = self.Gait_Net.test_1(intermediates)
            all_outs.append(outs)

        embed_list, log_list = self.Gait_Net.test_2(
                                        torch.cat(all_outs, dim=2),
                                        seqL,
                                        )
        
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
                    'image/human_mask': self.min_max_norm(human_mask.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()),
                },
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},

                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                }
            }
        return retval
