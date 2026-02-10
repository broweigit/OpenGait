import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ...modules import SetBlockWrapper, SeparateFCs, SeparateBNNecks, PackSequenceWrapper, HorizontalPoolingPyramid
from torch.nn import functional as F

# ######################################## GaitBase ###########################################

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class AttentionFusion(nn.Module): 
    def __init__(self, in_channels, squeeze_ratio, feat_len):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.feat_len = feat_len
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * feat_len, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv3x3(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels * feat_len), 
            )
        )
    
    def forward(self, feat_list): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
            ...
        '''
        feats = torch.cat(feat_list, dim=1)
        score = self.conv(feats) # [n, 2 * c, s, h, w]
        score = rearrange(score, 'n (c d) s h w -> n c d s h w', d=self.feat_len)
        score = F.softmax(score, dim=2)
        retun = feat_list[0]*score[:,:,0]
        for i in range(1, self.feat_len):
            retun += feat_list[i]*score[:,:,i]
        return retun

from typing import Optional, Callable
class BasicBlock_Time(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.temb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256,planes),
        )

    def forward(self, x, temb, use_Time=True):
        identity = x

        out = self.conv1(x)

        if temb is not None and use_Time:
            out = out + self.temb_proj(temb)[:,:,None,None]

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ...modules import BasicConv2d
block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck,
             'BasicBlock_Time': BasicBlock_Time,}

# class Pre_ResNet9(ResNet):
#     def __init__(self, type, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
#         if block in block_map.keys():
#             block = block_map[block]
#         else:
#             raise ValueError(
#                 "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
#         self.maxpool_flag = maxpool
#         super(Pre_ResNet9, self).__init__(block, layers)

#         # Not used #
#         self.fc = None
#         self.layer2 = None
#         self.layer3 = None
#         self.layer4 = None
#         ############
#         self.inplanes = channels[0]
#         self.bn1 = nn.BatchNorm2d(self.inplanes)

#         self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

#         self.layer1 = self._make_layer(
#             block, channels[0], layers[0], stride=strides[0], dilate=False)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         if blocks >= 1:
#             layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
#         else:
#             def layer(x): return x
#         return layer

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         if self.maxpool_flag:
#             x = self.maxpool(x)

#         x = self.layer1(x)
#         return x

# class Post_ResNet9(ResNet):
#     def __init__(self, type, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
#         if block in block_map.keys():
#             block = block_map[block]
#         else:
#             raise ValueError(
#                 "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
#         super(Post_ResNet9, self).__init__(block, layers)
#         # Not used #
#         self.fc = None
#         self.conv1 = None
#         self.bn1 = None
#         self.relu = None
#         self.layer1 = None
#         ############
#         self.inplanes = channels[0]
#         self.layer2 = self._make_layer(
#             block, channels[1], layers[1], stride=strides[1], dilate=False)
#         self.layer3 = self._make_layer(
#             block, channels[2], layers[2], stride=strides[2], dilate=False)
#         self.layer4 = self._make_layer(
#             block, channels[3], layers[3], stride=strides[3], dilate=False)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         if blocks >= 1:
#             layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
#         else:
#             def layer(x): return x
#         return layer

#     def forward(self, x):
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x

class FlexibleSequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            try:
                # 尝试传递额外的参数和关键字参数
                input = module(input, *args, **kwargs)
            except TypeError:
                # 如果模块不需要额外参数，回退到仅传递输入
                input = module(input)
        return input


class Pre_ResNet9(ResNet):
    def __init__(self, type, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True, in_groups=1):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(Pre_ResNet9, self).__init__(BasicBlock, layers)

        # Not used #
        self.fc = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1, groups=in_groups)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        # return layer
        return FlexibleSequential(*list(layer.children()))

    def forward(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x, *args, **kwargs)
        return x

class Post_ResNet9(ResNet):
    def __init__(self, type, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True, in_groups=1):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        super(Post_ResNet9, self).__init__(BasicBlock, layers)
        # Not used #
        self.fc = None
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.layer1 = None
        ############
        self.inplanes = channels[0]
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        # return layer
        return FlexibleSequential(*list(layer.children()))

    def forward(self, x, *args, **kwargs):
        x = self.layer2(x, *args, **kwargs)
        x = self.layer3(x, *args, **kwargs)
        x = self.layer4(x, *args, **kwargs)
        return x


from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from ... import backbones
class Baseline(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline, self).__init__()
        model_cfg['backbone_cfg']['in_channel'] = model_cfg['Denoising_Branch']['target_dim']
        self.pre_part = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))

        model_cfg['backbone_cfg']['in_channel'] = model_cfg['Appearance_Branch']['target_dim']
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))

        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.fusion = AttentionFusion(**model_cfg['AttentionFusion'])

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def vis_forward(self, denosing, appearance, seqL):
        denosing = self.pre_part(denosing)  # [n, c, s, h, w]
        appearance = self.pre_rgb(appearance)  # [n, c, s, h, w]
        outs = self.fusion([denosing, appearance])
        return denosing, appearance, outs

    def forward(self, denosing, appearance, seqL):
        denosing = self.pre_part(denosing)  # [n, c, s, h, w]
        appearance = self.pre_rgb(appearance)  # [n, c, s, h, w]
        outs = self.fusion([denosing, appearance])
        # heat_mapt = rearrange(outs, 'n c s h w -> n s h w c')
        del denosing, appearance
        outs = self.post_backbone(outs)

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        
        # Horizontal Pooling Matching, HPM
        outs = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(outs)  # [n, c, p]
        _, logits = self.BNNecks(embed_1)  # [n, c, p]
        # return embed_1, logits, heat_mapt
        return embed_1, logits


class Baseline_Single(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_Single, self).__init__()
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def pre_forward(self, appearance, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)  # [n, c, s, h, w]
        outs = self.post_backbone(outs, *args, **kwargs)
        return outs

    def forward(self, appearance, seqL, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)  # [n, c, s, h, w]
        outs = self.post_backbone(outs, *args, **kwargs)
        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        outs = self.HPP(outs)  # [n, c, p]
        embed_1 = self.FCs(outs)  # [n, c, p]
        _, logits = self.BNNecks(embed_1)  # [n, c, p]
        return embed_1, logits
    
    def test_1(self, appearance, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)  # [n, c, s, h, w]
        outs = self.post_backbone(outs, *args, **kwargs)
        return outs

    def test_2(self, outs, seqL):
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]
        embed_1 = self.FCs(outs)  # [n, c, p]
        _, logits = self.BNNecks(embed_1)  # [n, c, p]
        return embed_1, logits
    
from ...modules import SemanticPartPooling, TemporalMotionAggregator
class Baseline_Part_Single(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_Part_Single, self).__init__()
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        # 🌟 1. 初始化新模块
        self.SPP = SemanticPartPooling(geo_order=2) # 开启5通道几何矩
        self.TMA = TemporalMotionAggregator()
        # 🌟 2. 调整 FC 输入维度 = 原始特征通道 + 5 (几何通道)
        self.parts_num = 7 
        in_c = model_cfg['SeparateFCs']['in_channels'] + 5 
        out_c = model_cfg['SeparateFCs']['out_channels']
        # ==================== 🌟 修复部分 Start ====================
        # 处理 SeparateFCs 的参数冲突
        fc_cfg = model_cfg['SeparateFCs'].copy()
        # 移除冲突键，防止 **fc_cfg 解包时与位置参数打架
        fc_cfg.pop('in_channels', None)
        fc_cfg.pop('out_channels', None)
        fc_cfg.pop('parts_num', None)
        
        self.FCs = SeparateFCs(self.parts_num, in_c, out_c, **fc_cfg)
        
        # 处理 SeparateBNNecks 的参数冲突
        # SeparateBNNecks 定义: (parts_num, in_channels, class_num, ...)
        bn_cfg = model_cfg['SeparateBNNecks'].copy()
        bn_cfg.pop('in_channels', None) # 这里的 in_channels 对应上面的 out_c
        bn_cfg.pop('parts_num', None)
        
        # 注意: bn_cfg 里面应该包含 'class_num'，这里直接解包即可
        self.BNNecks = SeparateBNNecks(self.parts_num, out_c, **bn_cfg)
        # ==================== 🌟 修复部分 End ====================

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def pre_forward(self, appearance, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)  # [n, c, s, h, w]
        outs = self.post_backbone(outs, *args, **kwargs)
        return outs

    def forward(self, appearance, seqL, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)  # [n, c, s, h, w]
        outs = self.post_backbone(outs, *args, **kwargs)
        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        outs = self.HPP(outs)  # [n, c, p]
        embed_1 = self.FCs(outs)  # [n, c, p]
        _, logits = self.BNNecks(embed_1)  # [n, c, p]
        return embed_1, logits
    
    def test_1(self, appearance, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)  # [n, c, s, h, w]
        outs = self.post_backbone(outs, *args, **kwargs)
        return outs

    # 🌟 3. 重写 test_2
    def test_2(self, x, seqL, parts_mask):
        # x: [n, c, s, h, w]
        # parts_mask: [n, s, 6, h, w]
        
        # Step 1: 空间-几何提取 (Space) -> [n, c+5, s, 6]
        part_feats_seq = self.SPP(x, parts_mask)
        
        # Step 2: 时序动态聚合 (Time) -> [n, c+5, 6]
        # 这里的 embed_1 包含了纹理特征和 (速度, 角速度, 形变速率)
        embed_1 = self.TMA(part_feats_seq) 
        
        # Step 3: 映射与分类
        # FC 层会自动学习几何特征与纹理特征的非线性组合
        embed_1 = self.FCs(embed_1) 
        _, logits = self.BNNecks(embed_1)
        
        return embed_1, logits
    
import math
def get_timestep_embedding(timesteps, embedding_dim, max_timesteps=40, frequency_scaling=10):
    """
    Adjusted sinusoidal embeddings for smaller timestep ranges.
    Args:
        timesteps: 1D tensor of shape (batch_size,)
        embedding_dim: Target embedding dimension
        max_timesteps: Maximum timestep value (e.g., 40)
        frequency_scaling: Scaling factor for frequency range (e.g., 10)
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    # 调整频率范围
    emb = math.log(max_timesteps * frequency_scaling) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Baseline_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        self.Gait_Net_1 = Baseline_Single(model_cfg)
        self.Gait_Net_2 = Baseline_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN//2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN//2)]
        )
        # self.Gait_List = nn.ModuleList([
        #     self.Gait_Net for _ in range(self.num_FPN)
        # ])

    def forward(self, x, seqL):
        x = self.test_1(x)
        embed_list, log_list = self.test_2(x, seqL)
        return embed_list, log_list

    def test_1(self, x, *args, **kwargs):
        # x: [n, c, s, h, w]
        n,c,s,h,w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        for i in range(self.num_FPN):
            x_list[i] = self.Gait_List[i].test_1(x_list[i], *args, **kwargs)
        x = torch.concat(x_list, dim=1)
        return x

    def test_2(self, x, seqL):
        # x: [n, c, s, h, w]
        # embed_1: [n, c, p]
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            embed_1, logits = self.Gait_List[i].test_2(x_list[i], seqL)
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list
    
class Baseline_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_ShareTime_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        self.Gait_Net_1 = Baseline_Single(model_cfg)
        self.Gait_Net_2 = Baseline_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN//2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN//2)]
        )
        # self.Gait_List = nn.ModuleList([
        #     self.Gait_Net for _ in range(self.num_FPN)
        # ])

        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def forward(self, x, seqL):
        x = self.test_1(x)
        embed_list, log_list = self.test_2(x, seqL)
        return embed_list, log_list

    def test_1(self, x, *args, **kwargs):
        # x: [n, c, s, h, w]
        n,c,s,h,w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1,-1).repeat(n*s,1)
        for i in range(self.num_FPN):
            
            temb = get_timestep_embedding(t[:,i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)

            x_list[i] = self.Gait_List[i].test_1(x_list[i], temb=temb, *args, **kwargs)
        x = torch.concat(x_list, dim=1)
        return x

    def test_2(self, x, seqL):
        # x: [n, c, s, h, w]
        # embed_1: [n, c, p]
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            embed_1, logits = self.Gait_List[i].test_2(x_list[i], seqL)
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list
    
class Baseline_Part_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_Part_ShareTime_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        # 只改这里
        self.Gait_Net_1 = Baseline_Part_Single(model_cfg)
        self.Gait_Net_2 = Baseline_Part_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN//2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN//2)]
        )
        # self.Gait_List = nn.ModuleList([
        #     self.Gait_Net for _ in range(self.num_FPN)
        # ])

        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def forward(self, x, seqL):
        x = self.test_1(x)
        embed_list, log_list = self.test_2(x, seqL)
        return embed_list, log_list

    def test_1(self, x, *args, **kwargs):
        # x: [n, c, s, h, w]
        n,c,s,h,w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1,-1).repeat(n*s,1)
        for i in range(self.num_FPN):
            
            temb = get_timestep_embedding(t[:,i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)

            x_list[i] = self.Gait_List[i].test_1(x_list[i], temb=temb, *args, **kwargs)
        x = torch.concat(x_list, dim=1)
        return x

    def test_2(self, x, seqL, parts_mask): # 新增 parts_mask
        # x: [n, c, s, h, w]
        # parts_mask: [n, s, 6, h, w]
        
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            # 这里的 parts_mask 对所有 FPN 层是共用的
            embed_1, logits = self.Gait_List[i].test_2(x_list[i], seqL, parts_mask)
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list

class Baseline_Share(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_Share, self).__init__()
        self.head_num = model_cfg['head_num']
        self.num_FPN = model_cfg['total_layer_num'] // model_cfg['group_layer_num']
        self.real_gait = nn.ModuleList([
            Baseline_Single(model_cfg) for _ in range(self.head_num)
        ])
        self.Gait_List = nn.ModuleList([
            self.real_gait[_ // (self.num_FPN // self.head_num)] for _ in range(self.num_FPN)
        ])

    def forward(self, x, seqL):
        x = self.test_1(x)
        embed_list, log_list = self.test_2(x, seqL)
        return embed_list, log_list

    def test_1(self, x, *args, **kwargs):
        # x: [n, c, s, h, w]
        n,c,s,h,w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        for i in range(self.num_FPN):
            x_list[i] = self.Gait_List[i].test_1(x_list[i], *args, **kwargs)
        x = torch.concat(x_list, dim=1)
        return x

    def test_2(self, x, seqL):
        # x: [n, c, s, h, w]
        # embed_1: [n, c, p]
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            embed_1, logits = self.Gait_List[i].test_2(x_list[i], seqL)
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list

class Baseline_Semantic_2B(Baseline_ShareTime_2B):
    """
    A variant of Baseline_ShareTime_2B that uses Semantic Attention Pooling 
    instead of Horizontal Pyramid Pooling (HPP).
    
    Logic:
    1. Input: Features [N, C, S, H, W] & Attention Maps [N, P, S, H, W]
    2. Per-Frame Semantic Pooling: Weighted Sum using Attn Map -> [N, C, S, P]
    3. Temporal Pooling: Aggregate over S -> [N, C, P]
    4. Heads: FCs & BNNecks
    """
    def __init__(self, model_cfg):
        super(Baseline_Semantic_2B, self).__init__(model_cfg)
        # 初始化逻辑完全复用父类，因为它已经包含了我们需要的 sub-networks (Gait_List)
        # Gait_List 中的每个 Baseline_Single 包含了 FCs, BNNecks 和 TP 模块，我们可以直接借用

    def forward(self, x, attn_map, seqL):
        # 注意：这里签名变了，增加了 attn_map
        x = self.test_1(x)
        embed_list, log_list = self.test_2(x, attn_map, seqL)
        return embed_list, log_list

    def test_2(self, x, attn_map, seqL):
        """
        Args:
            x: [n, c_total, s, h, w] - FPN Features
            attn_map: [n, p, s, h, w] - Semantic Attention Maps (P=70)
            seqL: Sequence Lengths for TP
        """
        n, c_total, s, h, w = x.shape
        n_attn, p, s_attn, h_attn, w_attn = attn_map.shape
        
        # 简单校验维度
        assert n == n_attn and s == s_attn, "Feature and AttnMap batch/time dims mismatch"
        
        # 将 Feature 切分为 FPN Heads
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        
        embed_list = []
        log_list = []
        
        for i in range(self.num_FPN):
            # 1. 获取当前 FPN Head 的特征
            # feat: [n, c_sub, s, h, w]
            feat = x_list[i]
            
            # =======================================================
            # 🌟 Step 1: Frame-level Semantic Pooling
            # =======================================================
            # 我们需要对每一帧 (n*s) 进行加权求和
            # Feat: [B_total, C, HW]
            # Map:  [B_total, P, HW]
            
            # 展平 Batch*Time 和 Spatial 维度
            feat_flat = rearrange(feat, 'n c s h w -> (n s) c (h w)')
            map_flat = rearrange(attn_map, 'n p s h w -> (n s) p (h w)')
            
            # 语义聚合 (Weighted Sum)
            # 公式: Output = Feat @ Map^T
            # [N*S, P, HW] @ [N*S, HW, C] -> [N*S, P, C]
            # 结果含义: 每一帧图像中，P 个关键点对应的 C 维特征
            sp_feat = torch.matmul(map_flat, feat_flat.transpose(1, 2))
            
            # =======================================================
            # 🌟 Step 2: Temporal Pooling (TP)
            # =======================================================
            # 还原维度以进行 TP: [N, S, P, C] -> Permute to [N, C, S, P]
            # OpenGait 的 TP (PackSequenceWrapper) 默认在 dim=2 (序列维度) 上操作
            sp_feat = rearrange(sp_feat, '(n s) p c -> n c s p', n=n, s=s)
            
            # 调用 Baseline_Single 里的 TP 模块 (通常是 Max Pooling)
            # Input: [N, C, S, P], Output: [N, C, P]
            # 注意：这里的 P (Parts) 相当于原来的 H/W 空间维度，TP 对它不敏感，只聚合 S
            tp_feat = self.Gait_List[i].TP(sp_feat, seqL, options={"dim": 2})[0]
            
            # =======================================================
            # 🌟 Step 3: Classification Heads
            # =======================================================
            # 直接调用子网的 FC 和 BNNeck
            # Input: [N, C, P] -> Output: [N, C, P]
            embed_1 = self.Gait_List[i].FCs(tp_feat)
            
            # Input: [N, C, P] -> Output: [N, ClassNum, P]
            _, logits = self.Gait_List[i].BNNecks(embed_1)
            
            embed_list.append(embed_1)
            log_list.append(logits)
            
        return embed_list, log_list