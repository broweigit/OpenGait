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


class HorizontalWidthTokenPyramid(nn.Module):
    def __init__(self, bin_num=None, width_token_num=4):
        super(HorizontalWidthTokenPyramid, self).__init__()
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num
        self.width_token_num = width_token_num

    def forward(self, x):
        """
            x  : [n, c, s, h, w]
            ret: [n, c, s, p, k]
        """
        n, c, s, h, w = x.size()
        features = []
        for b in self.bin_num:
            if h % b != 0:
                raise ValueError(f"Feature height {h} must be divisible by bin size {b}.")
            part_h = h // b
            z = x.view(n, c, s, b, part_h, w)
            if part_h == 1:
                # Avoid materializing both mean and max when the part already
                # collapses to a single row, which otherwise doubles peak memory.
                z = z.squeeze(-2)
                scale_after_pool = 2.0
            else:
                z_mean = z.mean(-2)
                z_max = z.max(-2)[0]
                z = z_mean.add_(z_max)
                scale_after_pool = 1.0
            z = rearrange(z, 'n c s p w -> (n s p) c w').contiguous()
            z_avg = F.adaptive_avg_pool1d(z, self.width_token_num)
            z_max = F.adaptive_max_pool1d(z, self.width_token_num)
            z = z_avg.add_(z_max)
            if scale_after_pool != 1.0:
                z.mul_(scale_after_pool)
            z = rearrange(
                z, '(n s p) c k -> n c s p k',
                n=n, s=s, p=b, k=self.width_token_num
            ).contiguous()
            features.append(z)
        return torch.cat(features, dim=3)


class WidthTokenTemporalMixer(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(WidthTokenTemporalMixer, self).__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.pwconv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
            x  : [n, c, s, p, k]
            ret: [n, c, s, p, k]
        """
        n, c, s, p, k = x.shape
        z = rearrange(x, 'n c s p k -> (n p k) c s').contiguous()
        z = self.dwconv(z)
        z = self.bn(z)
        z = self.act(z)
        z = self.pwconv(z)
        z = rearrange(z, '(n p k) c s -> n c s p k', n=n, p=p, k=k).contiguous()
        return x + z


class WidthTokenAttentionPooling(nn.Module):
    def __init__(self, channels, hidden_ratio=4):
        super(WidthTokenAttentionPooling, self).__init__()
        hidden_dim = max(channels // hidden_ratio, 16)
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        """
            x  : [n, c, p, k]
            ret: [n, c, p]
        """
        n, c, p, k = x.shape
        z = rearrange(x, 'n c p k -> (n p) c k').contiguous()
        attn = self.score(z)
        attn = F.softmax(attn, dim=-1)
        pooled = (z * attn).sum(dim=-1)
        return rearrange(pooled, '(n p) c -> n c p', n=n, p=p).contiguous()

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

        self.vertical_pooling = model_cfg.get('vertical_pooling', False)

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
        # 🌟 新增：如果开启纵向池化，交换 H 和 W 的物理位置
        if self.vertical_pooling:
            outs = outs.transpose(2, 3).contiguous() # 变成 [n, c, w, h]
        # Horizontal Pooling Matching, HPM
        outs = self.HPP(outs)  # [n, c, p]
        embed_1 = self.FCs(outs)  # [n, c, p]
        _, logits = self.BNNecks(embed_1)  # [n, c, p]
        return embed_1, logits
    
    def test_1(self, appearance, return_debug=False, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)  # [n, c, s, h, w]
        if not return_debug:
            outs = self.post_backbone(outs, *args, **kwargs)
            return outs

        n, c, s, h, w = outs.shape
        layer1_feat = outs.contiguous()
        x = outs.transpose(1, 2).reshape(-1, c, h, w)
        post_block = self.post_backbone.forward_block
        layer2_feat = post_block.layer2(x, *args, **kwargs)
        layer3_feat = post_block.layer3(layer2_feat, *args, **kwargs)
        layer4_feat = post_block.layer4(layer3_feat, *args, **kwargs)

        final_outs = layer4_feat.reshape(n, s, *layer4_feat.shape[1:]).transpose(1, 2).contiguous()
        layer2_feat = layer2_feat.reshape(n, s, *layer2_feat.shape[1:]).transpose(1, 2).contiguous()
        layer3_feat = layer3_feat.reshape(n, s, *layer3_feat.shape[1:]).transpose(1, 2).contiguous()
        return final_outs, {
            'layer1_feat': layer1_feat,
            'layer2_feat': layer2_feat,
            'layer3_feat': layer3_feat,
            'layer4_feat': final_outs,
        }

    def test_2(self, outs, seqL):
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        if self.vertical_pooling:
            outs = outs.transpose(2, 3).contiguous()
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


class HardPartPyramidPooling(nn.Module):
    def __init__(self, parts_num):
        super(HardPartPyramidPooling, self).__init__()
        self.parts_num = parts_num

    def forward(self, x, part_labels):
        n, c, s, h, w = x.shape
        if part_labels.dim() != 4:
            raise ValueError(f"part_labels should be [n, s, h, w], got {part_labels.shape}")

        if part_labels.shape[:2] != (n, s):
            raise ValueError(
                f"part_labels batch/time dims {part_labels.shape[:2]} do not match feature dims {(n, s)}"
            )

        if part_labels.shape[-2:] != (h, w):
            part_labels = F.interpolate(
                part_labels.float().view(n * s, 1, *part_labels.shape[-2:]),
                size=(h, w),
                mode='nearest'
            ).view(n, s, h, w).long()
        else:
            part_labels = part_labels.long()

        feat = rearrange(x, 'n c s h w -> (n s) c (h w)').contiguous()
        part_labels = rearrange(part_labels, 'n s h w -> (n s) (h w)').contiguous()

        if part_labels.min() < 0 or part_labels.max() >= self.parts_num:
            raise ValueError(
                f"part_labels should be in [0, {self.parts_num - 1}], got [{part_labels.min().item()}, {part_labels.max().item()}]"
            )

        ns = feat.shape[0]
        label_index = part_labels.unsqueeze(1).expand(-1, c, -1)

        pooled_sum = feat.new_zeros(ns, c, self.parts_num)
        pooled_sum.scatter_add_(2, label_index, feat)

        pooled_count = feat.new_zeros(ns, 1, self.parts_num)
        pooled_count.scatter_add_(
            2,
            part_labels.unsqueeze(1),
            feat.new_ones(ns, 1, feat.shape[-1])
        )

        pooled_max = feat.new_full((ns, c, self.parts_num), -100.0)
        pooled_max.scatter_reduce_(2, label_index, feat, reduce='amax', include_self=True)

        valid_mask = pooled_count > 0
        pooled_mean = pooled_sum / pooled_count.clamp_min(1.0)
        pooled_max = torch.where(valid_mask.expand(-1, c, -1), pooled_max, torch.zeros_like(pooled_max))

        pooled = pooled_mean + pooled_max
        return rearrange(pooled, '(n s) c p -> n c s p', n=n, s=s).contiguous()


class AnchorPatchPooling(nn.Module):
    def __init__(self, parts_num):
        super(AnchorPatchPooling, self).__init__()
        self.parts_num = parts_num

    def forward(self, feats, part_labels, valid_mask=None):
        if feats.dim() != 3:
            raise ValueError(f"feats should be [n, c, k], got {feats.shape}")

        n, c, k = feats.shape
        if part_labels.dim() != 1 or part_labels.numel() != k:
            raise ValueError(
                f"part_labels should be [k] and match anchor dim {k}, got {part_labels.shape}"
            )

        part_labels = part_labels.long()
        if part_labels.min() < 0 or part_labels.max() >= self.parts_num:
            raise ValueError(
                f"part_labels should be in [0, {self.parts_num - 1}], got "
                f"[{part_labels.min().item()}, {part_labels.max().item()}]"
            )

        if valid_mask is None:
            valid_mask = torch.ones(n, k, dtype=torch.bool, device=feats.device)
        elif valid_mask.shape != (n, k):
            raise ValueError(f"valid_mask should be {(n, k)}, got {valid_mask.shape}")
        else:
            valid_mask = valid_mask.bool()

        label_index = part_labels.view(1, 1, k).expand(n, c, k)
        label_index_count = part_labels.view(1, 1, k).expand(n, 1, k)

        valid_feat = feats * valid_mask.unsqueeze(1).to(feats.dtype)

        pooled_sum = feats.new_zeros(n, c, self.parts_num)
        pooled_sum.scatter_add_(2, label_index, valid_feat)

        pooled_count = feats.new_zeros(n, 1, self.parts_num)
        pooled_count.scatter_add_(
            2,
            label_index_count,
            valid_mask.unsqueeze(1).to(feats.dtype)
        )

        pooled_max = feats.new_full((n, c, self.parts_num), -100.0)
        pooled_max.scatter_reduce_(2, label_index, feats, reduce='amax', include_self=True)

        patch_count = feats.new_zeros(n, 1, self.parts_num)
        patch_count.scatter_add_(
            2,
            label_index_count,
            feats.new_ones(n, 1, k)
        )

        pooled_mean = pooled_sum / pooled_count.clamp_min(1.0)
        pooled_max = torch.where(
            patch_count.expand(-1, c, -1) > 0,
            pooled_max,
            torch.zeros_like(pooled_max)
        )

        return pooled_mean + pooled_max


class DynamicAnchorPartPooling(nn.Module):
    def __init__(self, parts_num):
        super(DynamicAnchorPartPooling, self).__init__()
        self.parts_num = parts_num

    def forward(self, feats, part_labels, valid_mask=None):
        if feats.dim() != 4:
            raise ValueError(f"feats should be [n, c, s, k], got {feats.shape}")

        n, c, s, k = feats.shape
        if part_labels.shape != (n, s, k):
            raise ValueError(
                f"part_labels should be {(n, s, k)}, got {part_labels.shape}"
            )

        part_labels = rearrange(part_labels.long(), 'n s k -> (n s) k').contiguous()
        if part_labels.min() < 0 or part_labels.max() >= self.parts_num:
            raise ValueError(
                f"part_labels should be in [0, {self.parts_num - 1}], got "
                f"[{part_labels.min().item()}, {part_labels.max().item()}]"
            )

        feats = rearrange(feats, 'n c s k -> (n s) c k').contiguous()
        ns = feats.shape[0]

        if valid_mask is None:
            valid_mask = torch.ones(ns, k, dtype=torch.bool, device=feats.device)
        elif valid_mask.shape != (n, s, k):
            raise ValueError(f"valid_mask should be {(n, s, k)}, got {valid_mask.shape}")
        else:
            valid_mask = rearrange(valid_mask.bool(), 'n s k -> (n s) k').contiguous()

        label_index = part_labels.unsqueeze(1).expand(-1, c, -1)
        label_index_count = part_labels.unsqueeze(1)

        valid_feat = feats * valid_mask.unsqueeze(1).to(feats.dtype)

        pooled_sum = feats.new_zeros(ns, c, self.parts_num)
        pooled_sum.scatter_add_(2, label_index, valid_feat)

        pooled_count = feats.new_zeros(ns, 1, self.parts_num)
        pooled_count.scatter_add_(
            2,
            label_index_count,
            valid_mask.unsqueeze(1).to(feats.dtype)
        )

        pooled_max = feats.new_full((ns, c, self.parts_num), -100.0)
        pooled_max.scatter_reduce_(2, label_index, feats, reduce='amax', include_self=True)

        patch_count = feats.new_zeros(ns, 1, self.parts_num)
        patch_count.scatter_add_(
            2,
            label_index_count,
            feats.new_ones(ns, 1, k)
        )

        pooled_mean = pooled_sum / pooled_count.clamp_min(1.0)
        pooled_max = torch.where(
            patch_count.expand(-1, c, -1) > 0,
            pooled_max,
            torch.zeros_like(pooled_max)
        )

        pooled = pooled_mean + pooled_max
        return rearrange(pooled, '(n s) c p -> n c s p', n=n, s=s).contiguous()


class Baseline_PartPP_Single(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_PartPP_Single, self).__init__()
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.parts_num = model_cfg['SeparateFCs']['parts_num']
        self.PartPP = HardPartPyramidPooling(self.parts_num)
        self.TP = PackSequenceWrapper(torch.max)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    def pre_forward(self, appearance, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        outs = self.post_backbone(outs, *args, **kwargs)
        return outs

    def forward(self, appearance, seqL, part_labels, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        outs = self.post_backbone(outs, *args, **kwargs)
        embed_1, logits = self.test_2(outs, seqL, part_labels)
        return embed_1, logits

    def test_1(self, appearance, return_debug=False, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        if not return_debug:
            outs = self.post_backbone(outs, *args, **kwargs)
            return outs

        n, c, s, h, w = outs.shape
        x = outs.transpose(1, 2).reshape(-1, c, h, w)
        post_block = self.post_backbone.forward_block
        layer2_feat = post_block.layer2(x, *args, **kwargs)
        x = post_block.layer3(layer2_feat, *args, **kwargs)
        x = post_block.layer4(x, *args, **kwargs)

        final_outs = x.reshape(n, s, *x.shape[1:]).transpose(1, 2).contiguous()
        layer2_feat = layer2_feat.reshape(n, s, *layer2_feat.shape[1:]).transpose(1, 2).contiguous()
        return final_outs, {'layer2_feat': layer2_feat}

    def test_2(self, outs, seqL, part_labels):
        part_feats = self.PartPP(outs, part_labels)
        embed_1 = self.TP(part_feats, seqL, options={"dim": 2})[0]
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

    def test_1(self, x, return_debug=False, *args, **kwargs):
        # x: [n, c, s, h, w]
        n,c,s,h,w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1,-1).repeat(n*s,1)
        debug_layer_feats = {
            'layer1_feat': [],
            'layer2_feat': [],
            'layer3_feat': [],
            'layer4_feat': [],
        }
        for i in range(self.num_FPN):
            
            temb = get_timestep_embedding(t[:,i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)

            outputs = self.Gait_List[i].test_1(
                x_list[i], return_debug=return_debug, temb=temb, *args, **kwargs
            )
            if return_debug:
                x_list[i], debug_info = outputs
                for key in debug_layer_feats.keys():
                    debug_layer_feats[key].append(debug_info[key])
            else:
                x_list[i] = outputs
        x = torch.concat(x_list, dim=1)
        if return_debug:
            debug_info = {
                key: torch.concat(feats, dim=1)
                for key, feats in debug_layer_feats.items()
            }
            for key, feats in debug_layer_feats.items():
                debug_info[f"{key}_list"] = feats
            return x, debug_info
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


class LatentFactorPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_num,
        horizontal_stripes=1,
        latent_per_stripe=None,
        support_kernel_size=3,
        temperature=0.125,
        presence_topk_ratio=0.125,
        active_gate_threshold=0.05,
        eps=1.0e-6,
    ):
        super(LatentFactorPooling, self).__init__()
        self.latent_num = latent_num
        self.horizontal_stripes = max(1, int(horizontal_stripes))
        if latent_per_stripe is None:
            if latent_num % self.horizontal_stripes != 0:
                raise ValueError(
                    f"latent_num={latent_num} must be divisible by horizontal_stripes={self.horizontal_stripes}."
                )
            latent_per_stripe = latent_num // self.horizontal_stripes
        self.latent_per_stripe = int(latent_per_stripe)
        if self.horizontal_stripes * self.latent_per_stripe != latent_num:
            raise ValueError(
                "horizontal_stripes * latent_per_stripe must equal latent_num, got "
                f"{self.horizontal_stripes} * {self.latent_per_stripe} != {latent_num}."
            )
        self.competition_latent_num = self.latent_per_stripe
        self.support_kernel_size = support_kernel_size
        self.temperature = temperature
        self.presence_topk_ratio = presence_topk_ratio
        self.active_gate_threshold = active_gate_threshold
        self.eps = eps
        latent_basis = torch.zeros(
            self.horizontal_stripes, self.latent_per_stripe, in_channels
        )
        nn.init.xavier_uniform_(latent_basis.view(-1, in_channels))
        self.latent_basis = nn.Parameter(latent_basis)

    def _build_active_mask(self, x):
        feat_energy = x.detach().float().pow(2).mean(dim=1)
        feat_energy = feat_energy / feat_energy.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        active_mask = (feat_energy > self.active_gate_threshold).float()

        fallback_mask = (feat_energy > 0).float()
        use_fallback = active_mask.sum(dim=(-2, -1), keepdim=True) <= 0
        active_mask = torch.where(use_fallback, fallback_mask, active_mask)

        active_mask = rearrange(active_mask, 'n h w -> n (h w)').contiguous()
        feat_energy = rearrange(feat_energy, 'n h w -> n (h w)').contiguous()
        return active_mask, feat_energy

    def forward(self, x, return_debug=False, return_aux=False):
        if x.dim() != 4:
            raise ValueError(f"LatentFactorPooling expects [n, c, h, w], got {x.shape}")

        n, c, h, w = x.shape
        stripe_num = self.horizontal_stripes
        if h % stripe_num != 0:
            raise ValueError(
                f"Feature height {h} must be divisible by horizontal_stripes={stripe_num}."
            )
        stripe_h = h // stripe_num

        stripe_feat = rearrange(
            x, 'n c (g hs) w -> (n g) c hs w', g=stripe_num, hs=stripe_h
        ).contiguous()
        support_feat = F.avg_pool2d(
            stripe_feat,
            kernel_size=self.support_kernel_size,
            stride=1,
            padding=self.support_kernel_size // 2,
        )

        feat_flat = rearrange(
            stripe_feat, '(n g) c hs w -> n g (hs w) c', n=n, g=stripe_num, hs=stripe_h, w=w
        ).contiguous()
        support_flat = rearrange(
            support_feat, '(n g) c hs w -> n g (hs w) c', n=n, g=stripe_num, hs=stripe_h, w=w
        ).contiguous()

        active_mask, active_energy = self._build_active_mask(stripe_feat)
        active_mask = rearrange(
            active_mask, '(n g) p -> n g p', n=n, g=stripe_num
        ).contiguous()
        active_energy = rearrange(
            active_energy, '(n g) p -> n g p', n=n, g=stripe_num
        ).contiguous()
        support_flat = F.normalize(support_flat.float(), p=2, dim=-1)
        latent_basis = F.normalize(self.latent_basis.float(), p=2, dim=-1)
        support_logits = torch.einsum(
            'ngpc,gkc->ngpk', support_flat, latent_basis
        ) / self.temperature
        route_prob = F.softmax(support_logits, dim=-1)
        active_mask = active_mask.to(route_prob.dtype)
        support = route_prob * active_mask.unsqueeze(-1)

        pool_weights = support / support.sum(dim=2, keepdim=True).clamp_min(self.eps)
        tokens = torch.einsum('ngpk,ngpc->ngkc', pool_weights, feat_flat.float())
        tokens = rearrange(tokens, 'n g k c -> n c (g k)').contiguous()

        position_count = support.shape[2]
        topk = max(1, min(position_count, int(position_count * self.presence_topk_ratio)))
        presence = support.topk(topk, dim=2).values.mean(dim=2)
        presence = rearrange(presence, 'n g k -> n (g k)').contiguous()
        presence = presence / presence.sum(dim=1, keepdim=True).clamp_min(self.eps)

        support_map_local = rearrange(
            support, 'n g (hs w) k -> n g k hs w', hs=stripe_h, w=w
        ).contiguous()
        pool_weights_local = rearrange(
            pool_weights, 'n g (hs w) k -> n g k hs w', hs=stripe_h, w=w
        ).contiguous()
        support_map = support.new_zeros(n, stripe_num, self.latent_per_stripe, h, w)
        pool_weights_map = pool_weights.new_zeros(n, stripe_num, self.latent_per_stripe, h, w)
        for stripe_idx in range(stripe_num):
            h_start = stripe_idx * stripe_h
            h_end = h_start + stripe_h
            support_map[:, stripe_idx, :, h_start:h_end, :] = support_map_local[:, stripe_idx]
            pool_weights_map[:, stripe_idx, :, h_start:h_end, :] = pool_weights_local[:, stripe_idx]
        support_map = rearrange(support_map, 'n g k h w -> n (g k) h w').contiguous()
        pool_weights_map = rearrange(pool_weights_map, 'n g k h w -> n (g k) h w').contiguous()
        support_flat_global = rearrange(support_map, 'n k h w -> n (h w) k').contiguous()
        pool_weights_flat_global = rearrange(
            pool_weights_map, 'n k h w -> n (h w) k'
        ).contiguous()
        active_mask_map = rearrange(
            active_mask, 'n g (hs w) -> n 1 (g hs) w', hs=stripe_h, w=w
        ).contiguous()
        active_energy_map = rearrange(
            active_energy, 'n g (hs w) -> n 1 (g hs) w', hs=stripe_h, w=w
        ).contiguous()

        if not return_debug and not return_aux:
            return tokens.to(x.dtype), presence.to(x.dtype)

        aux_info = {
            'active_mask_flat': rearrange(
                active_mask_map, 'n 1 h w -> n (h w)'
            ).contiguous().to(x.dtype),
            'active_energy_flat': rearrange(
                active_energy_map, 'n 1 h w -> n (h w)'
            ).contiguous().to(x.dtype),
            'support_flat': support_flat_global.to(x.dtype),
            'pool_weights_flat': pool_weights_flat_global.to(x.dtype),
        }
        if return_debug:
            aux_info.update({
                'support_map': support_map.to(x.dtype),
                'pool_weights': pool_weights_map.to(x.dtype),
                'active_mask': active_mask_map.to(x.dtype),
                'active_energy': active_energy_map.to(x.dtype),
            })
        return tokens.to(x.dtype), presence.to(x.dtype), aux_info


class Baseline_LatentSet_Single(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_LatentSet_Single, self).__init__()
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.TP = PackSequenceWrapper(torch.max)
        self.TP_mean = PackSequenceWrapper(torch.mean)
        self.vertical_pooling = model_cfg.get('vertical_pooling', False)

        latent_cfg = model_cfg.get('latent_cfg', {})
        latent_reg_cfg = model_cfg.get('latent_reg_cfg', {})
        latent_bin_num = model_cfg['bin_num']
        latent_num = int(latent_bin_num) if isinstance(latent_bin_num, int) else sum(latent_bin_num)
        expected_parts = model_cfg['SeparateFCs']['parts_num']
        if latent_num != expected_parts:
            raise ValueError(
                f"Latent branch expects sum(bin_num) == SeparateFCs.parts_num, got {latent_num} vs {expected_parts}."
            )

        in_channels = model_cfg['SeparateFCs']['in_channels']
        self.LatentPool = LatentFactorPooling(
            in_channels=in_channels,
            latent_num=latent_num,
            horizontal_stripes=latent_cfg.get('horizontal_stripes', 1),
            latent_per_stripe=latent_cfg.get('latent_per_stripe', None),
            support_kernel_size=latent_cfg.get('support_kernel_size', 3),
            temperature=latent_cfg.get('temperature', 0.125),
            presence_topk_ratio=latent_cfg.get('presence_topk_ratio', 0.125),
            active_gate_threshold=latent_cfg.get('active_gate_threshold', 0.05),
        )
        self.latent_reg_cfg = {
            'basis_orth_weight': float(latent_reg_cfg.get('basis_orth_weight', 0.0)),
            'token_decor_weight': float(latent_reg_cfg.get('token_decor_weight', 0.0)),
            'overlap_weight': float(latent_reg_cfg.get('overlap_weight', 0.0)),
            'balance_weight': float(latent_reg_cfg.get('balance_weight', 0.0)),
            'coverage_weight': float(latent_reg_cfg.get('coverage_weight', 0.0)),
            'support_competition_weight': float(latent_reg_cfg.get('support_competition_weight', 0.0)),
            'support_balance_weight': float(latent_reg_cfg.get('support_balance_weight', 0.0)),
            'support_overlap_weight': float(latent_reg_cfg.get('support_overlap_weight', 0.0)),
        }
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    @staticmethod
    def _offdiag_square_mean(gram):
        if gram.dim() == 2:
            gram = gram.unsqueeze(0)
        k = gram.size(-1)
        if k <= 1:
            return gram.new_tensor(0.0)
        eye = torch.eye(k, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        offdiag = gram * (1.0 - eye)
        return offdiag.pow(2).sum(dim=(-2, -1)).mean() / (k * (k - 1))

    def _compute_latent_regularization(self, tokens, token_weights, aux_info):
        reg_losses = {}
        reg_cfg = self.latent_reg_cfg
        if all(weight <= 0 for weight in reg_cfg.values()):
            return reg_losses

        support_flat = aux_info['support_flat'].float()
        active_mask = aux_info['active_mask_flat'].float()
        support_prob = support_flat / support_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if reg_cfg['basis_orth_weight'] > 0:
            latent_basis = F.normalize(self.LatentPool.latent_basis.float(), p=2, dim=-1)
            basis_gram = latent_basis.matmul(latent_basis.transpose(1, 2))
            reg_losses['latent_reg/basis_orth'] = (
                self._offdiag_square_mean(basis_gram) * reg_cfg['basis_orth_weight']
            )

        if reg_cfg['token_decor_weight'] > 0:
            token_feat = F.normalize(tokens.float().transpose(1, 2), p=2, dim=-1)
            token_gram = token_feat.matmul(token_feat.transpose(1, 2))
            reg_losses['latent_reg/token_decor'] = (
                self._offdiag_square_mean(token_gram) * reg_cfg['token_decor_weight']
            )

        if reg_cfg['support_competition_weight'] > 0:
            entropy_norm = torch.log(torch.tensor(
                float(max(self.LatentPool.competition_latent_num, 2)),
                device=support_prob.device,
                dtype=support_prob.dtype,
            )).clamp_min(1e-6)
            support_entropy = -(support_prob * support_prob.clamp_min(1e-6).log()).sum(dim=-1)
            support_entropy = support_entropy / entropy_norm
            active_count = active_mask.sum(dim=1).clamp_min(1.0)
            support_entropy = (support_entropy * active_mask).sum(dim=1) / active_count
            reg_losses['latent_reg/support_competition'] = (
                support_entropy.mean() * reg_cfg['support_competition_weight']
            )

        if reg_cfg['support_balance_weight'] > 0:
            mean_support_prob = (
                support_prob * active_mask.unsqueeze(-1)
            ).sum(dim=(0, 1)) / active_mask.sum().clamp_min(1.0)
            target_support_prob = torch.full_like(
                mean_support_prob, 1.0 / mean_support_prob.numel()
            )
            reg_losses['latent_reg/support_balance'] = (
                F.mse_loss(mean_support_prob, target_support_prob) * reg_cfg['support_balance_weight']
            )

        if reg_cfg['support_overlap_weight'] > 0:
            masked_support_prob = support_prob * active_mask.unsqueeze(-1)
            masked_support_prob = masked_support_prob / masked_support_prob.norm(
                p=2, dim=1, keepdim=True
            ).clamp_min(1e-6)
            support_gram = masked_support_prob.transpose(1, 2).matmul(masked_support_prob)
            reg_losses['latent_reg/support_overlap'] = (
                self._offdiag_square_mean(support_gram) * reg_cfg['support_overlap_weight']
            )

        if reg_cfg['overlap_weight'] > 0:
            pool_weights = aux_info['pool_weights_flat'].float()
            pool_weights = pool_weights / pool_weights.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
            overlap_gram = pool_weights.transpose(1, 2).matmul(pool_weights)
            reg_losses['latent_reg/spatial_overlap'] = (
                self._offdiag_square_mean(overlap_gram) * reg_cfg['overlap_weight']
            )

        if reg_cfg['balance_weight'] > 0:
            mean_presence = token_weights.float().mean(dim=0)
            target_presence = torch.full_like(mean_presence, 1.0 / mean_presence.numel())
            reg_losses['latent_reg/usage_balance'] = (
                F.mse_loss(mean_presence, target_presence) * reg_cfg['balance_weight']
            )

        if reg_cfg['coverage_weight'] > 0:
            active_target = active_mask
            active_target = active_target / active_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
            coverage_map = aux_info['pool_weights_flat'].float().sum(dim=-1)
            coverage_map = coverage_map / coverage_map.sum(dim=1, keepdim=True).clamp_min(1e-6)
            reg_losses['latent_reg/coverage'] = (
                F.l1_loss(coverage_map, active_target) * reg_cfg['coverage_weight']
            )

        return reg_losses

    def pre_forward(self, appearance, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        outs = self.post_backbone(outs, *args, **kwargs)
        return outs

    def forward(self, appearance, seqL, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        outs = self.post_backbone(outs, *args, **kwargs)
        embed, logits, token_weights = self.test_2(outs, seqL)
        return embed, logits, token_weights

    def test_1(self, appearance, return_debug=False, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        if not return_debug:
            outs = self.post_backbone(outs, *args, **kwargs)
            return outs

        n, c, s, h, w = outs.shape
        x = outs.transpose(1, 2).reshape(-1, c, h, w)
        post_block = self.post_backbone.forward_block
        layer2_feat = post_block.layer2(x, *args, **kwargs)
        x = post_block.layer3(layer2_feat, *args, **kwargs)
        x = post_block.layer4(x, *args, **kwargs)

        final_outs = x.reshape(n, s, *x.shape[1:]).transpose(1, 2).contiguous()
        layer2_feat = layer2_feat.reshape(n, s, *layer2_feat.shape[1:]).transpose(1, 2).contiguous()
        return final_outs, {'layer2_feat': layer2_feat}

    def test_2(self, outs, seqL, return_debug=False, return_reg=False):
        if self.vertical_pooling:
            outs = outs.transpose(3, 4).contiguous()

        n, c, s, h, w = outs.shape
        frame_outs = rearrange(outs, 'n c s h w -> (n s) c h w').contiguous()

        need_aux = return_debug or return_reg
        if need_aux:
            frame_tokens, frame_token_weights, aux_info = self.LatentPool(
                frame_outs, return_debug=return_debug, return_aux=True
            )
        else:
            frame_tokens, frame_token_weights = self.LatentPool(frame_outs, return_debug=False)

        token_seq = rearrange(frame_tokens, '(n s) c k -> n c s k', n=n, s=s).contiguous()
        token_weight_seq = rearrange(frame_token_weights, '(n s) k -> n k s', n=n, s=s).contiguous()

        tokens = self.TP(token_seq, seqL, options={"dim": 2})[0]
        token_weights = self.TP_mean(token_weight_seq, seqL, options={"dim": 2})
        embed = self.FCs(tokens)
        _, logits = self.BNNecks(embed)

        reg_losses = self._compute_latent_regularization(
            frame_tokens, frame_token_weights, aux_info
        ) if return_reg else {}

        if return_debug:
            debug_info = {
                'support_map': rearrange(
                    aux_info['support_map'],
                    '(n s) k h w -> n s k h w',
                    n=n, s=s,
                ).contiguous(),
                'pool_weights': rearrange(
                    aux_info['pool_weights'],
                    '(n s) k h w -> n s k h w',
                    n=n, s=s,
                ).contiguous(),
                'active_mask': rearrange(
                    aux_info['active_mask'],
                    '(n s) c h w -> n s c h w',
                    n=n, s=s,
                ).contiguous(),
                'active_energy': rearrange(
                    aux_info['active_energy'],
                    '(n s) c h w -> n s c h w',
                    n=n, s=s,
                ).contiguous(),
            }
            aux_info.update({
                'token_weights': token_weights,
                'frame_token_weights': rearrange(
                    frame_token_weights, '(n s) k -> n s k', n=n, s=s
                ).contiguous(),
                'frame_tokens': token_seq,
                'tokens': tokens,
                'embeddings': embed,
            })
            debug_info.update({
                'token_weights': token_weights,
                'frame_token_weights': aux_info['frame_token_weights'],
                'frame_tokens': aux_info['frame_tokens'],
                'tokens': tokens,
                'embeddings': embed,
            })
            if return_reg:
                return embed, logits, token_weights, reg_losses, debug_info
            return embed, logits, token_weights, debug_info
        if return_reg:
            return embed, logits, token_weights, reg_losses
        return embed, logits, token_weights


class Baseline_LatentSet_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_LatentSet_ShareTime_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        self.Gait_Net_1 = Baseline_LatentSet_Single(model_cfg)
        self.Gait_Net_2 = Baseline_LatentSet_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN // 2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN // 2)]
        )

        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def forward(self, x, seqL):
        x = self.test_1(x)
        embed_list, log_list, weight_list = self.test_2(x, seqL)
        return embed_list, log_list, weight_list

    def test_1(self, x, return_debug=False, *args, **kwargs):
        n, c, s, h, w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1, -1).repeat(n * s, 1)
        debug_layer2_list = []
        for i in range(self.num_FPN):
            temb = get_timestep_embedding(t[:, i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)

            outputs = self.Gait_List[i].test_1(
                x_list[i], return_debug=return_debug, temb=temb, *args, **kwargs
            )
            if return_debug:
                x_list[i], debug_info = outputs
                debug_layer2_list.append(debug_info['layer2_feat'])
            else:
                x_list[i] = outputs

        x = torch.concat(x_list, dim=1)
        if return_debug:
            return x, {'layer2_feat': torch.concat(debug_layer2_list, dim=1)}
        return x

    def test_2(self, x, seqL, return_debug=False, return_reg=False):
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        weight_list = []
        reg_list = []
        debug_list = []
        for i in range(self.num_FPN):
            outputs = self.Gait_List[i].test_2(
                x_list[i], seqL, return_debug=return_debug, return_reg=return_reg
            )
            if return_debug:
                if return_reg:
                    embed_1, logits, token_weights, reg_losses, debug_info = outputs
                    reg_list.append(reg_losses)
                else:
                    embed_1, logits, token_weights, debug_info = outputs
                debug_list.append(debug_info)
            else:
                if return_reg:
                    embed_1, logits, token_weights, reg_losses = outputs
                    reg_list.append(reg_losses)
                else:
                    embed_1, logits, token_weights = outputs
            embed_list.append(embed_1)
            log_list.append(logits)
            weight_list.append(token_weights)

        if return_debug:
            if return_reg:
                return embed_list, log_list, weight_list, reg_list, debug_list
            return embed_list, log_list, weight_list, debug_list
        if return_reg:
            return embed_list, log_list, weight_list, reg_list
        return embed_list, log_list, weight_list


class Baseline_HPPWidthToken_Single(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_HPPWidthToken_Single, self).__init__()
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.WidthTokenPyramid = HorizontalWidthTokenPyramid(
            bin_num=model_cfg['bin_num'],
            width_token_num=model_cfg.get('width_token_num', 4)
        )
        self.WidthTemporal = WidthTokenTemporalMixer(
            channels=model_cfg['SeparateFCs']['in_channels'],
            kernel_size=model_cfg.get('width_temporal_kernel_size', 3)
        )
        self.WidthPool = WidthTokenAttentionPooling(
            channels=model_cfg['SeparateFCs']['in_channels'],
            hidden_ratio=model_cfg.get('width_attention_hidden_ratio', 4)
        )
        self.vertical_pooling = model_cfg.get('vertical_pooling', False)
        expected_parts = model_cfg['SeparateFCs']['parts_num']
        actual_parts = sum(model_cfg['bin_num'])
        if actual_parts != expected_parts:
            raise ValueError(
                f"Width-token branch expects sum(bin_num) == SeparateFCs.parts_num, "
                f"got {actual_parts} vs {expected_parts}."
            )

    def test_1(self, appearance, return_debug=False, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        if not return_debug:
            outs = self.post_backbone(outs, *args, **kwargs)
            return outs

        n, c, s, h, w = outs.shape
        x = outs.transpose(1, 2).reshape(-1, c, h, w)
        post_block = self.post_backbone.forward_block
        layer2_feat = post_block.layer2(x, *args, **kwargs)
        x = post_block.layer3(layer2_feat, *args, **kwargs)
        x = post_block.layer4(x, *args, **kwargs)

        final_outs = x.reshape(n, s, *x.shape[1:]).transpose(1, 2).contiguous()
        layer2_feat = layer2_feat.reshape(n, s, *layer2_feat.shape[1:]).transpose(1, 2).contiguous()
        return final_outs, {'layer2_feat': layer2_feat}

    def test_2(self, outs, seqL, return_debug=False):
        if self.vertical_pooling:
            outs = outs.transpose(3, 4).contiguous()
        outs = self.WidthTokenPyramid(outs)
        outs = self.WidthTemporal(outs)
        widthtoken_feat = self.TP(outs, seqL, options={"dim": 2})[0]
        pre_fc_feat = self.WidthPool(widthtoken_feat)
        embed_1 = self.FCs(pre_fc_feat)
        _, logits = self.BNNecks(embed_1)
        if return_debug:
            return embed_1, logits, {
                'pre_fc_feat': pre_fc_feat,
                'widthtoken_feat': widthtoken_feat,
            }
        return embed_1, logits


class Baseline_HPPWidthToken_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_HPPWidthToken_ShareTime_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        self.Gait_Net_1 = Baseline_HPPWidthToken_Single(model_cfg)
        self.Gait_Net_2 = Baseline_HPPWidthToken_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN // 2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN // 2)]
        )

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

    def test_1(self, x, return_debug=False, *args, **kwargs):
        n, c, s, h, w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1, -1).repeat(n * s, 1)
        debug_layer2_list = []
        for i in range(self.num_FPN):
            temb = get_timestep_embedding(t[:, i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)
            outputs = self.Gait_List[i].test_1(
                x_list[i], return_debug=return_debug, temb=temb, *args, **kwargs
            )
            if return_debug:
                x_list[i], debug_info = outputs
                debug_layer2_list.append(debug_info['layer2_feat'])
            else:
                x_list[i] = outputs
        final_outs = torch.concat(x_list, dim=1)
        if return_debug:
            return final_outs, {'layer2_feat': torch.concat(debug_layer2_list, dim=1)}
        return final_outs

    def test_2(self, x, seqL, return_debug=False):
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        debug_list = []
        for i in range(self.num_FPN):
            outputs = self.Gait_List[i].test_2(x_list[i], seqL, return_debug=return_debug)
            if return_debug:
                embed_1, logits, debug_info = outputs
                debug_list.append(debug_info)
            else:
                embed_1, logits = outputs
            embed_list.append(embed_1)
            log_list.append(logits)
        if return_debug:
            return embed_list, log_list, debug_list
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


class Baseline_PartPP_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_PartPP_ShareTime_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        self.Gait_Net_1 = Baseline_PartPP_Single(model_cfg)
        self.Gait_Net_2 = Baseline_PartPP_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN // 2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN // 2)]
        )

        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def forward(self, x, seqL, part_labels):
        x = self.test_1(x)
        embed_list, log_list = self.test_2(x, seqL, part_labels)
        return embed_list, log_list

    def test_1(self, x, return_debug=False, *args, **kwargs):
        n, c, s, h, w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1, -1).repeat(n * s, 1)
        debug_layer2_list = []
        for i in range(self.num_FPN):
            temb = get_timestep_embedding(t[:, i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)
            outputs = self.Gait_List[i].test_1(
                x_list[i], return_debug=return_debug, temb=temb, *args, **kwargs
            )
            if return_debug:
                x_list[i], debug_info = outputs
                debug_layer2_list.append(debug_info['layer2_feat'])
            else:
                x_list[i] = outputs
        final_outs = torch.concat(x_list, dim=1)
        if return_debug:
            return final_outs, {'layer2_feat': torch.concat(debug_layer2_list, dim=1)}
        return final_outs

    def test_2(self, x, seqL, part_labels):
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            embed_1, logits = self.Gait_List[i].test_2(x_list[i], seqL, part_labels)
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list


class Baseline_AnchorMasked_Single(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_AnchorMasked_Single, self).__init__()
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.TP = PackSequenceWrapper(torch.max)
        self.parts_num = model_cfg['SeparateFCs']['parts_num']
        self.PatchPool = AnchorPatchPooling(self.parts_num)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    def test_1(self, appearance, return_debug=False, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        if not return_debug:
            outs = self.post_backbone(outs, *args, **kwargs)
            return outs

        n, c, s, h, w = outs.shape
        x = outs.transpose(1, 2).reshape(-1, c, h, w)
        post_block = self.post_backbone.forward_block
        layer2_feat = post_block.layer2(x, *args, **kwargs)
        x = post_block.layer3(layer2_feat, *args, **kwargs)
        x = post_block.layer4(x, *args, **kwargs)

        final_outs = x.reshape(n, s, *x.shape[1:]).transpose(1, 2).contiguous()
        layer2_feat = layer2_feat.reshape(n, s, *layer2_feat.shape[1:]).transpose(1, 2).contiguous()
        return final_outs, {'layer2_feat': layer2_feat}

    def _temporal_any(self, valid_mask, seqL):
        if valid_mask is None:
            return None

        if seqL is None:
            return valid_mask.any(dim=1)

        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + torch.tensor(seqL).cumsum(0).tolist()[:-1]

        pooled_valid = []
        for curr_start, curr_seqL in zip(start, seqL):
            seq_valid = valid_mask.narrow(1, curr_start, curr_seqL)
            pooled_valid.append(seq_valid.any(dim=1))
        return torch.cat(pooled_valid, dim=0)

    def test_2(self, anchor_feats, seqL, anchor_valid=None, anchor_part_labels=None):
        if anchor_part_labels is None:
            raise ValueError("anchor_part_labels is required for anchor patch pooling.")

        pooled_anchor = self.TP(anchor_feats, seqL, options={"dim": 2})[0]
        pooled_valid = self._temporal_any(anchor_valid, seqL)
        embed_1 = self.PatchPool(pooled_anchor, anchor_part_labels, pooled_valid)
        embed_1 = self.FCs(embed_1)
        _, logits = self.BNNecks(embed_1)
        return embed_1, logits


class Baseline_AnchorMasked_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_AnchorMasked_ShareTime_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        self.Gait_Net_1 = Baseline_AnchorMasked_Single(model_cfg)
        self.Gait_Net_2 = Baseline_AnchorMasked_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN // 2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN // 2)]
        )

        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def test_1(self, x, return_debug=False, *args, **kwargs):
        n, c, s, h, w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1, -1).repeat(n * s, 1)
        debug_layer2_list = []
        for i in range(self.num_FPN):
            temb = get_timestep_embedding(t[:, i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)
            outputs = self.Gait_List[i].test_1(
                x_list[i], return_debug=return_debug, temb=temb, *args, **kwargs
            )
            if return_debug:
                x_list[i], debug_info = outputs
                debug_layer2_list.append(debug_info['layer2_feat'])
            else:
                x_list[i] = outputs
        final_outs = torch.concat(x_list, dim=1)
        if return_debug:
            return final_outs, {'layer2_feat': torch.concat(debug_layer2_list, dim=1)}
        return final_outs

    def test_2(self, anchor_feats, seqL, anchor_valid=None, anchor_part_labels=None):
        feat_list = torch.chunk(anchor_feats, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            embed_1, logits = self.Gait_List[i].test_2(
                feat_list[i], seqL, anchor_valid, anchor_part_labels
            )
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list


class Baseline_AnchorHeightPart_Single(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_AnchorHeightPart_Single, self).__init__()
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.TP = PackSequenceWrapper(torch.max)
        self.parts_num = model_cfg['SeparateFCs']['parts_num']
        self.DynamicPartPool = DynamicAnchorPartPooling(self.parts_num)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    def test_1(self, appearance, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        outs = self.post_backbone(outs, *args, **kwargs)
        return outs

    def test_2(self, anchor_feats, seqL, anchor_valid=None, anchor_part_labels=None):
        if anchor_part_labels is None:
            raise ValueError("anchor_part_labels is required for dynamic height-part pooling.")

        pooled_parts = self.DynamicPartPool(anchor_feats, anchor_part_labels, anchor_valid)
        embed_1 = self.TP(pooled_parts, seqL, options={"dim": 2})[0]
        embed_1 = self.FCs(embed_1)
        _, logits = self.BNNecks(embed_1)
        return embed_1, logits


class Baseline_AnchorHeightPart_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline_AnchorHeightPart_ShareTime_2B, self).__init__()
        self.num_FPN = model_cfg['num_FPN']
        self.Gait_Net_1 = Baseline_AnchorHeightPart_Single(model_cfg)
        self.Gait_Net_2 = Baseline_AnchorHeightPart_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN // 2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN // 2)]
        )

        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def test_1(self, x, *args, **kwargs):
        n, c, s, h, w = x.shape
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        t = torch.tensor(list(range(self.num_FPN))).to(x).view(1, -1).repeat(n * s, 1)
        for i in range(self.num_FPN):
            temb = get_timestep_embedding(t[:, i], self.t_channel, max_timesteps=self.num_FPN).to(x)
            temb = self.temb_proj(temb)
            x_list[i] = self.Gait_List[i].test_1(x_list[i], temb=temb, *args, **kwargs)
        return torch.concat(x_list, dim=1)

    def test_2(self, anchor_feats, seqL, anchor_valid=None, anchor_part_labels=None):
        feat_list = torch.chunk(anchor_feats, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            embed_1, logits = self.Gait_List[i].test_2(
                feat_list[i], seqL, anchor_valid, anchor_part_labels
            )
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
