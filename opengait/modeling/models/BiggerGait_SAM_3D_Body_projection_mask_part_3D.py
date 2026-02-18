import sys
import os
import roma
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from functools import partial

# import GaitBase
from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import save_image, pca_image
from ..modules import GaitAlign

# =========================================================================
# Helper Functions
# =========================================================================

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
                nn.Linear(source_dim//2, target_dim),
                )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim//2),
                    nn.BatchNorm1d(source_dim//2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim//2, source_dim),
                    )
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

import torch.nn.functional as F

class GeometryOptimalTransport(nn.Module):
    def __init__(self, temperature=0.01, dist_thresh=0.2, num_iters=3):
        super().__init__()
        self.epsilon = temperature
        self.dist_thresh = dist_thresh
        self.num_iters = num_iters # 迭代次数少一点(3次)，即为"软"Sinkhorn

    def forward(self, source_feats, source_locs, target_locs, source_valid_mask=None, target_valid_mask=None):
        """
        Args:
            source_feats: [B, N, C]
            source_locs:  [B, N, 2]
            target_locs:  [B, M, 2]
            source_valid_mask: [B, N]
            target_valid_mask: [B, M]
        """
        B, N, C = source_feats.shape
        M = target_locs.shape[1]
        
        # 1. 计算代价矩阵 Cost (同老方法)
        diff = target_locs.unsqueeze(2) - source_locs.unsqueeze(1)
        dist_sq = torch.sum(diff ** 2, dim=-1)

        # 2. 构建 Log-Kernel (同老方法，但在 Log 域)
        # Log_K_ij = -C_ij / epsilon
        log_K = -dist_sq / (self.epsilon + 1e-8)

        # 3. 处理 Mask (同老方法，逻辑一致)
        valid_connection = dist_sq < (self.dist_thresh ** 2)
        if source_valid_mask is not None:
            valid_connection = valid_connection & source_valid_mask.unsqueeze(1)
        if target_valid_mask is not None:
            valid_connection = valid_connection & target_valid_mask.unsqueeze(2)
        
        # 填充 -1e9 (Log 域的 0)
        log_K = log_K.masked_fill(~valid_connection, -1e9)

        # ==========================================================
        # 4. Sinkhorn 迭代 (Log-Domain)
        # 这里的改进：我们只迭代 3 次，这是一种"部分 OT"。
        # 它比 Softmax 更锐利，但又不像完全收敛的 OT 那样死板（允许一定的质量不平衡）。
        # ==========================================================
        
        # 初始化势能
        v = torch.zeros(B, 1, N, device=source_feats.device) # Source 势能
        u = torch.zeros(B, M, 1, device=source_feats.device) # Target 势能

        for _ in range(self.num_iters):
            # 步骤 A: Target 归一化 (类似 Softmax 的行归一化)
            # u = -logsumexp(log_K + v)
            # 这一步保证了每个 Target 像素能"抢"到足够的特征
            u = -torch.logsumexp(log_K + v, dim=2, keepdim=True)
            
            # 步骤 B: Source 归一化 (列归一化)
            # v = -logsumexp(log_K + u)
            # 这一步抑制了被过度复用的 Source 像素
            v = -torch.logsumexp(log_K + u, dim=1, keepdim=True)
            
            # 【关键修正】：防止 v 在全是 Mask 的列变成 inf
            # 如果某列 Source 全是无效连接，logsumexp 结果是 -inf，v 变成 inf
            # 我们需要把这些无效列的 v 重置为 0，防止污染后续计算
            if source_valid_mask is not None:
                v = v.masked_fill(~source_valid_mask.unsqueeze(1), 0.0)

        # 5. 计算最终 Attention Map
        # P = exp(log_K + u + v)
        log_P = log_K + u + v
        attn = torch.exp(log_P)
        
        # 再次硬过滤 (双重保险，同老方法)
        attn = attn * valid_connection.float()
        
        # ==========================================================
        # 6. 特征搬运 (同老方法)
        # ==========================================================
        target_feats = torch.bmm(attn, source_feats)

        # 7. 最终清理 (同老方法)
        if target_valid_mask is not None:
            target_feats = target_feats * target_valid_mask.unsqueeze(-1).float()
            
        # 检查是否有没有来源的目标点
        # 注意：在 Sinkhorn 中，u 会自动补偿，但如果真的没有任何连接，还是需要置 0
        has_source = valid_connection.any(dim=-1, keepdim=True)
        target_feats = target_feats * has_source.float()

        return target_feats

# =========================================================================
# Main Model
# =========================================================================

class BiggerGait__SAM3DBody__Projection_Mask_Part_3D_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        # 1. 基础参数
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]

        # 🌟 新增：加载 MHR 官方索引
        idx_path = 'pretrained_LVMs/mhr_part_indices.pt'
        if os.path.exists(idx_path):
            self.part_indices = torch.load(idx_path, map_location='cpu')
            self.msg_mgr.log_info(f"[MHR] Official part indices loaded from {idx_path}")
        else:
            self.msg_mgr.log_warning(f"[MHR] Warning: {idx_path} not found! Segmentation will fail.")
            self.part_indices = None

        # FPN Configuration
        layer_cfg = model_cfg.get("layer_config", {})
        self.layers_per_group = layer_cfg.get("layers_per_group", 2)
        
        # Hook Mask
        if "hook_mask" in layer_cfg:
            self.hook_mask = layer_cfg["hook_mask"]
        else:
            self.hook_mask = [False]*16 + [True]*16 # Default Top-16

        self.total_hooked_layers = sum(self.hook_mask)
        
        # Dimensions Check
        if self.total_hooked_layers % self.layers_per_group != 0:
            raise ValueError(f"Hook layers ({self.total_hooked_layers}) must be divisible by group size ({self.layers_per_group})")
        
        self.total_groups = self.total_hooked_layers // self.layers_per_group
        
        if self.total_groups % self.num_FPN != 0:
            raise ValueError(f"Total groups ({self.total_groups}) must be divisible by FPN heads ({self.num_FPN})")

        self.layers_per_head = self.total_hooked_layers // self.num_FPN
        input_dim = self.f4_dim * self.layers_per_head
        
        self.chunk_size = model_cfg.get("chunk_size", 96)

        # GaitNet & PreConv
        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)

        # FPN Heads
        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(input_dim, affine=False),
                nn.Conv2d(input_dim, self.f4_dim//2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size*2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])
        
        # Mask Branch (Keep structure but not used for projection logic)
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        
        self.init_SAM_Backbone()

        self.ot_solver = GeometryOptimalTransport(temperature=0.01, dist_thresh=0.2, num_iters=8)

    def init_SAM_Backbone(self):
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)
        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(f"Cannot import setup_sam_3d_body. Error: {e}")

        self.msg_mgr.log_info(f"[SAM3D] Loading SAM 3D Body (Encoder + Decoder)...")
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')
        
        self.SAM_Engine = estimator.model
        self.Backbone = self.SAM_Engine.backbone.encoder
        
        self.SAM_Engine.cpu() # 先放 CPU

        # ================= Hook Logic =================
        self.intermediate_features = {}
        self.hook_handles = []

        def get_activation(idx_in_list):
            def hook(model, input, output):
                if isinstance(output, (list, tuple)): output = output[0]
                if isinstance(output, (list, tuple)): output = output[0]
                self.intermediate_features[idx_in_list] = output
            return hook

        all_blocks = self.Backbone.blocks

        hook_count = 0
        for layer_idx, should_hook in enumerate(self.hook_mask):
            if should_hook:
                handle = all_blocks[layer_idx].register_forward_hook(get_activation(hook_count))
                self.hook_handles.append(handle)
                hook_count += 1
        
        self.msg_mgr.log_info(f"[SAM3D] Hooked {hook_count} layers.")
        
        # Freeze Everything
        self.SAM_Engine.eval()
        for param in self.SAM_Engine.parameters():
            param.requires_grad = False

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        # Re-initialize SAM (since modules() init might have messed it up)
        self.init_SAM_Backbone()
        
        self.SAM_Engine.eval()
        self.SAM_Engine.requires_grad_(False)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Model Count: {:.5f}M'.format(n_parameters / 1e6))

    # --- 🌟 Helper: Prepare Dummy Batch for SAM Decoder ---
    def _prepare_dummy_batch(self, image_embeddings, target_h, target_w):
        B = image_embeddings.shape[0]
        device = image_embeddings.device
        
        estimated_focal_length = max(target_h, target_w) * 1.1
        cx, cy = target_w / 2.0, target_h / 2.0
        
        cam_int = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3).clone()
        cam_int[:, 0, 0] = estimated_focal_length 
        cam_int[:, 1, 1] = estimated_focal_length 
        cam_int[:, 0, 2] = cx
        cam_int[:, 1, 2] = cy
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(target_h, device=device),
            torch.arange(target_w, device=device),
            indexing='ij'
        )
        ray_x = (x_grid - cx) / estimated_focal_length
        ray_y = (y_grid - cy) / estimated_focal_length
        ray_cond = torch.stack([ray_x, ray_y], dim=0).unsqueeze(0).expand(B, 2, target_h, target_w)

        bbox_scale = torch.tensor([max(target_h, target_w)], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 1)
        bbox_center = torch.tensor([cx, cy], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        img_size = torch.tensor([float(target_w), float(target_h)], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        affine_trans = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 2, 3)

        return {
            "img": torch.zeros(B, 1, 3, target_h, target_w, device=device),
            "ori_img_size": img_size, "img_size": img_size, "bbox_center": bbox_center,
            "bbox_scale": bbox_scale, "cam_int": cam_int, "affine_trans": affine_trans,
            "ray_cond": ray_cond, 
        }

    def get_source_vertex_index_map(self, vertices, cam_t, cam_int, H_feat, W_feat, target_H, target_W):
        """
        渲染 Index Map 和 Depth Map。
        Returns:
            index_map: [B, H, W] 像素对应的顶点索引
            depth_map: [B, 1, H, W] 像素对应的最小深度值
        """
        B, N_verts, _ = vertices.shape
        device = vertices.device
        
        # 1. 投影到原图相机平面
        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)
        
        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        
        u = (x / z_safe) * fx + cx 
        v = (y / z_safe) * fy + cy 
        
        # 2. 量化坐标
        u_feat = (u / target_W * W_feat).long().clamp(0, W_feat - 1)
        v_feat = (v / target_H * H_feat).long().clamp(0, H_feat - 1)
        flat_pixel_indices = v_feat * W_feat + u_feat # [B, N_verts]
        
        # 3. Z-Buffer：找出每个像素最近的深度
        depth_map_flat = torch.full((B, H_feat * W_feat), 1e6, device=device)
        depth_map_flat.scatter_reduce_(1, flat_pixel_indices, z, reduce='amin', include_self=False)
        
        # 4. 生成 Index Map
        min_depth_per_vertex = torch.gather(depth_map_flat, 1, flat_pixel_indices)
        is_visible = (z < (min_depth_per_vertex + 1e-4))
        
        index_map_flat = torch.full((B, H_feat * W_feat), -1, dtype=torch.long, device=device)
        vertex_indices = torch.arange(N_verts, device=device).unsqueeze(0).expand(B, -1)
        
        # 修正之前提到的 view 连续性问题，改用 reshape
        mask_flat = is_visible.reshape(-1) 
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * (H_feat * W_feat)
        global_pixel_indices = (flat_pixel_indices + batch_offsets).reshape(-1) 
        
        valid_pixel_indices = global_pixel_indices[mask_flat]
        valid_vertex_indices = vertex_indices.reshape(-1)[mask_flat]
        
        index_map_global = index_map_flat.reshape(-1)
        index_map_global[valid_pixel_indices] = valid_vertex_indices
        
        # 返回两个值：Index Map 和 Depth Map
        # 注意：Depth Map 恢复成 [B, 1, H, W] 以兼容你后续的 mask = (depth_map < 1e5).float()
        return index_map_global.reshape(B, H_feat, W_feat), depth_map_flat.reshape(B, 1, H_feat, W_feat)
    
    def warp_features_with_ot(self, human_feat, mask_src, 
                              pred_verts, pred_keypoints, pred_cam_t, global_rot, 
                              cam_int_src, cam_int_tgt, cam_t_tgt,
                              H_feat, W_feat, target_H, target_W):
        """
        利用最优传输将 human_feat 从原视角迁移到 90度侧视视角
        关键改进：显式生成 Target Mask，防止特征泄露到背景
        """
        B, C, H, W = human_feat.shape
        device = human_feat.device
        
        # =========================================================
        # 1. Source 端几何计算 (哪些 Source 像素有效？)
        # =========================================================
        # 获取 Source Index Map: [B, H, W] (值是 vertex_idx, -1 表示无效)
        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int_src, H_feat, W_feat, target_H, target_W
        )
        
        # Source 有效性掩码：既要在原图分割 Mask 内，又要能找到对应的 Vertex
        valid_src_mask = (mask_src.squeeze(1) > 0.5) & (src_idx_map >= 0) # [B, H, W]
        
        flat_human_feat = rearrange(human_feat, 'b c h w -> b (h w) c')
        flat_src_idx_map = src_idx_map.view(B, -1)
        flat_src_mask = valid_src_mask.view(B, -1)
        
        # Gather Source Vertices 3D Coords [B, HW, 3]
        # 注意：对于无效点 (-1)，我们用 0 号顶点暂代，反正后面 mask 会把它屏蔽掉
        flat_src_verts = torch.zeros((B, H*W, 3), device=device)
        safe_indices = flat_src_idx_map.clone()
        safe_indices[safe_indices < 0] = 0
        flat_src_verts = torch.gather(pred_verts, 1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        
        # =========================================================
        # 2. Target 端几何计算 (哪些 Target 像素有效？)
        # =========================================================
        # 为了得到 Target Mask，我们需要将 Mesh 真正旋转并投影一次
        
        midhip = (pred_keypoints[:, 9] + pred_keypoints[:, 10]) / 2.0
        centered_verts = pred_verts - midhip.unsqueeze(1) # [B, N, 3]
        
        # 构建旋转矩阵 (Current -> Canonical Side)
        rot_fix = global_rot.clone(); rot_fix[..., [0,1,2]] *= -1
        R_canon = roma.euler_to_rotmat("XYZ", rot_fix) 
        R_side = torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]], device=device).view(1,3,3).expand(B,3,3)
        # 复合旋转: R_comp @ v (在 SMPL 坐标系下)
        R_comp = torch.matmul(R_canon.transpose(1,2), R_side.transpose(1,2))
        
        # 执行旋转 (注意坐标系翻转：OpenCV -> SMPL -> Rotate -> OpenCV)
        v_tmp = centered_verts.clone(); v_tmp[...,[1,2]] *= -1 
        v_rot_smpl = torch.bmm(v_tmp, R_comp)
        v_rot_cv = v_rot_smpl.clone(); v_rot_cv[...,[1,2]] *= -1 
        
        # 关键步骤：投影生成 Target Mask
        # 我们复用 get_source_vertex_index_map，只为了拿 depth_map
        _, tgt_depth_map = self.get_source_vertex_index_map(
            v_rot_cv, cam_t_tgt, cam_int_tgt, H_feat, W_feat, target_H, target_W
        )
        # 生成 Target 有效性掩码 [B, HW]
        valid_tgt_mask = (tgt_depth_map.view(B, -1) < 1e5) 
        
        # =========================================================
        # 3. 构建 OT 坐标系 (都投影到 Target 2D 平面)
        # =========================================================
        
        # A. 计算 Source Points 在 Target 图上的 "期望落点" (Projected Source Locs)
        # 我们对 flat_src_verts (Source像素对应的3D点) 应用同样的旋转和投影
        src_centered = flat_src_verts - midhip.unsqueeze(1)
        src_tmp = src_centered.clone(); src_tmp[...,[1,2]] *= -1
        src_rot_smpl = torch.bmm(src_tmp, R_comp)
        src_rot_cv = src_rot_smpl.clone(); src_rot_cv[...,[1,2]] *= -1
        
        # 投影到 Target 2D
        v_cam_tgt = src_rot_cv + cam_t_tgt.unsqueeze(1)
        x, y, z = v_cam_tgt.unbind(-1)
        z = z.clamp(min=1e-3) # 深度保护
        
        fx, fy = cam_int_tgt[:,0,0].unsqueeze(1), cam_int_tgt[:,1,1].unsqueeze(1)
        cx, cy = cam_int_tgt[:,0,2].unsqueeze(1), cam_int_tgt[:,1,2].unsqueeze(1)
        u_tgt = (x / z) * fx + cx
        v_tgt = (y / z) * fy + cy
        
        # 归一化 Source Locs [-1, 1]
        u_norm = 2.0 * (u_tgt / target_W) - 1.0
        v_norm = 2.0 * (v_tgt / target_H) - 1.0
        projected_source_locs = torch.stack([u_norm, v_norm], dim=-1) # [B, HW, 2]
        
        # B. 构建 Target Grid Locs (实际网格坐标)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_feat, device=device),
            torch.linspace(-1, 1, W_feat, device=device),
            indexing='ij'
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, 2)
        
        # =========================================================
        # 4. 执行 OT (Attention)
        # =========================================================
        # 传入所有 Mask，确保只有 (有效Source -> 有效Target) 之间产生流量
        
        transported_feats = self.ot_solver(
            flat_human_feat, 
            projected_source_locs, 
            target_grid_locs, 
            source_valid_mask=flat_src_mask, # 屏蔽无效源
            target_valid_mask=valid_tgt_mask # 屏蔽背景目标 (关键！)
        )
        
        # 恢复形状
        warped_feat = rearrange(transported_feats, 'b (h w) c -> b c h w', h=H_feat)
        
        return warped_feat
    
    # def get_standard_90deg_projection(self, vertices, global_orient, h_feat, w_feat, target_h, target_w):
    #     """
    #     利用 SMPL Global Orientation 将点云还原到标准侧面视角
    #     Args:
    #         vertices: [B, N, 3] 相机坐标系下的点云
    #         global_orient: [B, 3] (轴角)
    #         h_feat, w_feat: 特征图尺寸
    #         target_h, target_w: 投影基准尺寸
    #     """
    #     B = vertices.shape[0]
    #     device = vertices.device
        
    #     # 1. 关键：修正坐标系翻转
    #     # MHR 输出时翻转了 Y 和 Z，我们先镜像回来，使其回到标准 SMPL 坐标空间进行计算
    #     v_fix = vertices.clone()
    #     v_fix[..., [1, 2]] *= -1
    #     # v_fix = v_fix - v_fix.mean(dim=1, keepdim=True) # 中心化到骨架中心

    #     # Debug (偏转角取反)
    #     global_orient_fix = global_orient.clone()
    #     global_orient_fix[..., [0, 1, 2]] *= -1
    #     R_full = roma.euler_to_rotmat("XYZ", global_orient_fix)

    #     # 2. 构造 "标准态 -> 侧身态" 的 90 度旋转矩阵
    #     # 对应 Canonical 空间下的绕 Y 轴旋转
    #     R_90 = torch.tensor([
    #         [ 0., 0., 1.],
    #         [ 0., 1., 0.],
    #         [-1., 0., 0.]
    #     ], device=device, dtype=vertices.dtype).view(1, 3, 3).expand(B, 3, 3)

    #     R_composite = torch.matmul(R_full.transpose(1, 2), R_90.transpose(1, 2))

    #     # 3. 执行旋转
    #     # 使用 bmm 将中心化后的点云旋转到标准图
    #     v_smpl = torch.bmm(v_fix, R_composite) # [B, N, 3]

    #     # 4. 关键：投影前镜像回 MHR 坐标系 (Y轴向下)
    #     # 这样 project_vertices_to_mask_and_depth 才能正确处理
    #     v_mhr = v_smpl.clone()
    #     v_mhr[..., [1, 2]] *= -1

    #     # 5. 标准虚拟相机参数 TODO
    #     # 根据经验，focal 设为 target_h * 1.1 比较稳妥
    #     focal = max(target_h, target_w) * 1.1
    #     cam_int_90 = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3).clone()
    #     cam_int_90[:, 0, 0] = focal
    #     cam_int_90[:, 1, 1] = focal
    #     cam_int_90[:, 0, 2] = target_w / 2.0
    #     cam_int_90[:, 1, 2] = target_h / 2.0
        
    #     # 将相机放在前方 2.2 米处
    #     cam_t_90 = torch.zeros((B, 3), device=device)
    #     cam_t_90[:, 2] = 2.2 

    #     # 6. 投影生成 90 度 Mask
    #     mask_90, _ = self.project_vertices_to_mask_and_depth(
    #         v_mhr, cam_t_90, cam_int_90, h_feat, w_feat, target_h, target_w
    #     )
    #     return mask_90

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        CHUNK_SIZE = self.chunk_size # e.g. 4
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_outs = []
        
        # 图像目标尺寸 (512, 256)
        target_h, target_w = self.image_size * 2, self.image_size 
        # 特征图尺寸 (32, 16) -> Mask 在这里生成
        h_feat, w_feat = target_h // 16, target_w // 16

        # 定义部位顺序 (必须与 SemanticPartPooling 中的逻辑一致)
        ordered_parts = ["head", "torso", "l_arm", "r_arm", "l_leg", "r_leg"]

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            # B(n*s), C, H, W
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            curr_bs = rgb_img.shape[0]
            
            with torch.no_grad():
                # 1. Resize & Backbone
                outs = self.preprocess(rgb_img, target_h, target_w)
                self.intermediate_features = {}
                _ = self.Backbone(outs)
                
                # 2. SAM Decoder Preparation
                last_hook_idx = len(self.hook_handles) - 1
                sam_emb = self.intermediate_features[last_hook_idx] # [B, 517, 1280]
                
                # DINOv3 (B,N(517),C) -> SAM (B,C,H,W)
                target_tokens = h_feat * w_feat # 512
                
                # 剔除 CLS/Registers，只保留 Spatial Tokens
                if sam_emb.shape[1] > target_tokens:
                    sam_emb = sam_emb[:, -target_tokens:, :] # [B, 512, 1280]
                
                # [B, N, C] -> [B, C, N] -> [B, C, H, W]
                sam_emb = sam_emb.transpose(1, 2).reshape(curr_bs, -1, h_feat, w_feat) # [B, 1280, 32, 16]
                
                # Prepare Inputs
                dummy_batch = self._prepare_dummy_batch(sam_emb, target_h, target_w)
                self.SAM_Engine._batch_size = curr_bs
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(curr_bs, device=rgb.device)
                self.SAM_Engine.hand_batch_idx = []
                cond_info = torch.zeros(curr_bs, 3, device=rgb.device); cond_info[:, 2] = 1.1
                dummy_kp = torch.zeros(curr_bs, 1, 3, device=rgb.device); dummy_kp[..., -1] = -2

                # 3. Run Decoder
                with torch.amp.autocast(enabled=False, device_type='cuda'):
                     _, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=sam_emb, 
                        keypoints=dummy_kp, 
                        condition_info=cond_info, 
                        batch=dummy_batch
                    )
                
                # 4. 生成 Mask (32x16)
                pred_verts = pose_outs[-1]['pred_vertices'] 
                pred_cam_t = pose_outs[-1]['pred_cam_t']
                cam_int = dummy_batch['cam_int']    
                
                # --- 遮挡去重逻辑 (Z-Buffer) ---
                part_depths = {}
                # 第一遍：收集每个部位的原始深度图
                if self.part_indices is not None:
                    for name, idxs in self.part_indices.items():
                        _, p_depth = self.get_source_vertex_index_map(
                            pred_verts[:, idxs, :], pred_cam_t, cam_int, 
                            h_feat, w_feat, target_h, target_w
                        )
                        part_depths[name] = p_depth

                    # 计算全局最小深度 [B, 1, 32, 16]
                    if len(part_depths) > 0:
                        all_depth_tensors = torch.cat(list(part_depths.values()), dim=1)
                        global_min_depth, _ = torch.min(all_depth_tensors, dim=1, keepdim=True)
                    else:
                        global_min_depth = torch.zeros((curr_bs, 1, h_feat, w_feat), device=rgb.device)

                    # 第二遍循环：生成去重后的 Mask
                    final_disjoint_masks = {}
                    part_summaries = {} # 用于 Visual Summary
                    part_colors = {
                        "head": [1.0, 0.0, 0.0], "torso": [0.0, 1.0, 1.0], 
                        "l_arm": [0.0, 1.0, 0.0], "r_arm": [1.0, 1.0, 0.0],
                        "l_leg": [0.0, 0.0, 1.0], "r_leg": [1.0, 0.0, 1.0]
                    }

                    for name in ordered_parts:
                        if name in part_depths:
                            p_depth = part_depths[name]
                            # 1. 深度竞争，获得基础硬掩码
                            is_closest = (p_depth == global_min_depth) & (p_depth < 1e5)
                            mask = is_closest.float()

                            # 2. 执行掩码膨胀 (Dilation)
                            # kernel_size=3 可以让每个点向四周扩充 1 像素
                            # mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                            
                            final_disjoint_masks[name] = mask # [B, 1, H, W]
                            
                            # 生成单部位叠加图 (仅限前5个样本)
                            if name in part_colors and curr_bs > 0:
                                m_high = F.interpolate(is_closest.float(), (target_h, target_w), mode='bilinear', align_corners=False)
                                c_vec = torch.tensor(part_colors[name], device=rgb.device).view(1, 3, 1, 1)
                                part_overlay = outs * 0.2 + (m_high * c_vec) * 0.8
                                part_summaries[f'image/part_{name}'] = part_overlay[:3].float()
                        else:
                            # 兜底：如果某个 part 没生成，给全 0
                            final_disjoint_masks[name] = torch.zeros((curr_bs, 1, h_feat, w_feat), device=rgb.device)

                    # 3. 堆叠 6 个部位 Mask [B, 6, H, W]
                    part_masks = torch.cat([final_disjoint_masks[k] for k in ordered_parts], dim=1)

                    # 4. 生成总 Mask 用于 FPN 前的背景滤除
                    generated_mask = torch.clamp(torch.sum(part_masks, dim=1, keepdim=True), 0, 1) # [B, 1, H, W]

                else:
                    raise RuntimeError("Part indices for MHR not loaded; cannot generate part masks.")

                # 5. 收集特征用于 FPN (Early Masking)
                mask_flat = generated_mask.view(curr_bs, -1, 1) # [B, 512, 1]
                
                features_to_use = []
                for i in range(len(self.hook_handles)):
                    feat = self.intermediate_features[i] 
                    if feat.shape[1] > target_tokens:
                        feat = feat[:, -target_tokens:, :] 
                    feat = feat * mask_flat # 应用总 Mask 去背景
                    features_to_use.append(feat)

            # =======================================================
            # 6. FPN Processing (Masked Features)
            # =======================================================
            processed_feat_list = []
            step = len(features_to_use) // self.num_FPN
            
            for i in range(self.num_FPN):
                sub_feats = features_to_use[i*step : (i+1)*step]
                
                # A. 拼接特征 [B, 512, C*step(4)]
                sub_app = torch.concat(sub_feats, dim=-1)
                sub_app = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim * len(sub_feats), elementwise_affine=False)(sub_app)
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                
                # D. FPN Head (HumanSpace_Conv)
                # 这一步包含 Conv + Upsample (ResizeToHW)
                reduced_feat = self.HumanSpace_Conv[i](sub_app) # [B, 64, 64, 32]
                processed_feat_list.append(reduced_feat)
                
                del sub_app, sub_feats

            # 7. 拼接 FPN 输出
            human_feat = torch.concat(processed_feat_list, dim=1) # [B, Total_C, 64, 32]

            # =======================================================
            # 🌟 [NEW] Geometry-Guided Feature Warping (OT)
            # =======================================================
            
            # A. 准备数据
            pred_verts = pose_outs[-1]['pred_vertices']
            global_rot = pose_outs[-1]['global_rot']
            pred_cam_t = pose_outs[-1]['pred_cam_t']
            pred_keypoints = pose_outs[-1]['pred_keypoints_3d']
            
            # 原图相机参数
            cam_int_src = dummy_batch['cam_int'] 
            
            # 生成原图的总 Mask (用于 OT 过滤无效点)
            # 这里的 part_masks 是你之前用原始投影生成的
            full_mask_src = F.interpolate(generated_mask.float(), (self.sils_size*2, self.sils_size), mode='bilinear', align_corners=False) # [B, 1, H, W]

            # B. 构建虚拟目标相机 (Canonical 90度侧视)
            # 焦距：1.1 * max(H, W)
            focal_tgt = max(target_h, target_w) * 1.1
            cx_tgt, cy_tgt = target_w / 2.0, target_h / 2.0
            
            cam_int_tgt = torch.eye(3, device=rgb.device).unsqueeze(0).expand(curr_bs, 3, 3).clone()
            cam_int_tgt[:, 0, 0] = focal_tgt
            cam_int_tgt[:, 1, 1] = focal_tgt
            cam_int_tgt[:, 0, 2] = cx_tgt
            cam_int_tgt[:, 1, 2] = cy_tgt
            
            # 目标平移：将人放在前方 2.2 米处 (Canonical Depth)
            cam_t_tgt = torch.zeros((curr_bs, 3), device=rgb.device)
            cam_t_tgt[:, 2] = 2.2 

            # C. 执行特征迁移
            # human_feat: [B, C, 64, 32] -> warped_feat: [B, C, 64, 32]
            # 注意：这里的特征是正侧面的！
            warped_feat = self.warp_features_with_ot(
                human_feat, 
                full_mask_src, 
                pred_verts, pred_keypoints, pred_cam_t, global_rot,
                cam_int_src, cam_int_tgt, cam_t_tgt,
                self.sils_size*2, self.sils_size, # Feat Size: 64, 32
                target_h, target_w                # Render Size: 512, 256
            )

            # C. Reshape for GaitNet [n, c, s, h, w]
            # 将 (n*s) 解开
            warped_feat = rearrange(warped_feat, '(n s) c h w -> n c s h w', n=n, s=s).contiguous()
            
            # 9. GaitNet Part 1 (ResNet)
            # Input:  [n, C, s, H, W]
            # Output: [n, C_out, s, H', W']
            outs = self.Gait_Net.test_1(warped_feat)

            all_outs.append(outs)

        # GaitNet Part 2 (时序聚合)
        embed_list, log_list = self.Gait_Net.test_2(
            torch.cat(all_outs, dim=2), # [n, c, s_chunk, h, w]
            seqL
        )
        
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.cat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.cat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:3].float(),
                    **part_summaries,
                    'image/generated_3d_mask_lowres': generated_mask.view(n*s, 1, h_feat, w_feat)[:3].float(),
                },
                'inference_feat': {
                    'embeddings': torch.cat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': torch.cat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        return retval