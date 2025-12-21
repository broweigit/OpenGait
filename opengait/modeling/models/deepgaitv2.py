import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D

from einops import rearrange

blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class DeepGaitV2(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        
        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        # # DEBUG: visualize sils and save
        # import matplotlib.pyplot as plt

        # # DEBUG: visualize sils and save
        # import os
        # import time
        # import numpy as np
        # import cv2
        # from PIL import Image as PILImage

        # # 1. 设置保存路径和唯一ID
        # debug_dir = './debug_vis_gifs'
        # os.makedirs(debug_dir, exist_ok=True)
        # # 使用当前毫秒时间戳作为唯一ID，防止覆盖
        # unique_id = int(time.time() * 1000) 

        # # 2. 获取第一个样本的数据 [C, S, H, W]
        # # sils shape: [n, c, s, h, w]
        # # 只取 Batch 0
        # if sils.shape[0] > 0:
        #     vis_data = sils[0].detach().cpu().numpy()
        #     C, S, H, W = vis_data.shape

        #     # === 打印统计信息 (Debug 关键) ===
        #     print(f"\n[DEBUG {unique_id}] Sils Shape: {sils.shape}")
        #     print(f"[DEBUG {unique_id}] Range: [{vis_data.min():.4f}, {vis_data.max():.4f}] | Mean: {vis_data.mean():.4f}")
            
        #     # 3. 归一化并转为 uint8 (0-255) 以便生成图像
        #     # 如果输入已经是 0-255 (Max > 1.0)，则直接转
        #     # 如果输入是 0-1 (Max <= 1.0)，则乘 255
        #     if vis_data.max() <= 1.0:
        #         vis_data = (vis_data * 255).astype(np.uint8)
        #     else:
        #         vis_data = vis_data.astype(np.uint8)

        #     frames = []
        #     # 4. 逐帧处理
        #     for t in range(S):
        #         # 准备 Bone (Channel 0) 和 Joint (Channel 1)
        #         # 如果只有1个通道，就只画那一个
        #         imgs_to_concat = []
        #         for c in range(C):
        #             img_gray = vis_data[c, t, :, :]
        #             # 应用伪彩色 (INFERNO 风格: 黑->紫->红->黄)
        #             img_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_INFERNO)
        #             # OpenCV 是 BGR，转为 RGB
        #             img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        #             imgs_to_concat.append(img_color)
                
        #         # 左右拼接: [Bone | Joint]
        #         if len(imgs_to_concat) > 1:
        #             # 中间加一条白线分割
        #             sep = np.ones((H, 5, 3), dtype=np.uint8) * 255
        #             canvas = np.concatenate([imgs_to_concat[0], sep, imgs_to_concat[1]], axis=1)
        #         else:
        #             canvas = imgs_to_concat[0]
                
        #         # 转为 PIL Image
        #         frames.append(PILImage.fromarray(canvas))

        #     # 5. 保存 GIF
        #     save_path = os.path.join(debug_dir, f'vis_{unique_id}_len{S}.gif')
        #     # duration=100ms (10fps)
        #     frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        #     print(f"[DEBUG {unique_id}] GIF saved to: {save_path}\n")

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
                embed = embed_2
        else:
                embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval
