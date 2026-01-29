import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from ..base_model import BaseModel
from .BigGait_utils.BigGait_GaitBase import *

# 引入 SAM 3D 依赖
sys.path.insert(0, "pretrained_LVMs/sam-3d-body")
# 注意：如果运行时提示找不到 notebook.utils，请确保你的路径结构正确
from notebook.utils import setup_sam_3d_body

def compute_pca_rgb(features):
    """
    简易 PCA 将高维特征降维到 3 RGB 通道用于可视化
    Args:
        features: [H, W, C]
    Returns:
        rgb: [H, W, 3] (Normalized 0-1)
    """
    shape = features.shape
    x = features.reshape(-1, shape[-1]) # [N_pixels, C]
    
    # 1. Center data
    mean = torch.mean(x, dim=0)
    x_centered = x - mean
    
    # 2. PCA using SVD
    _, _, V = torch.linalg.svd(x_centered.float(), full_matrices=False)
    components = V[1:4] # [3, C]
    projected = torch.matmul(x_centered.float(), components.T) # [N, 3]

    # 3. Normalize to [0, 1] for RGB
    p_min = projected.min(dim=0, keepdim=True)[0]
    p_max = projected.max(dim=0, keepdim=True)[0]
    rgb = (projected - p_min) / (p_max - p_min + 1e-6)
    
    return rgb.reshape(*shape[:-1], 3)

# =========================================================================
# 1. 核心模块: PointNetVLAD
# =========================================================================
class PointNetVLAD(nn.Module):
    def __init__(self, feature_size, num_clusters, ghost_clusters=0, alpha=10.0):
        super(PointNetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.feature_size = feature_size
        self.alpha = alpha
        
        # 1. Soft Assignment 卷积
        self.conv = nn.Conv1d(feature_size, num_clusters, kernel_size=1, bias=True)
        # 2. Cluster Centers
        self.centroids = nn.Parameter(torch.rand(num_clusters, feature_size))
        # 3. Init
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)**2
        )

    def forward(self, x):
        """
        Input: [B, C, N] -> Output: [B, C, K]
        """
        B, C, N = x.shape
        score = self.conv(x)
        soft_assign = F.softmax(score, dim=1) # [B, K, N]

        # VLAD Core
        # Term 1: [B, K, N] @ [B, N, C] -> [B, K, C]
        vlad_term1 = torch.matmul(soft_assign, x.transpose(1, 2))
        
        # Term 2: [B, K, 1] * [1, K, C] -> [B, K, C]
        sum_assign = soft_assign.sum(dim=-1, keepdim=True)
        vlad_term2 = sum_assign * self.centroids.unsqueeze(0)
        
        vlad = vlad_term1 - vlad_term2
        
        # Normalize
        vlad = F.normalize(vlad, p=2, dim=2, eps=1e-12)
        
        # Transpose to [B, C, K] for OpenGait
        vlad = vlad.transpose(1, 2)
        return vlad


# =========================================================================
# 2. 主模型: BiggerGait_SAM3D_NetVLAD
# =========================================================================
class BiggerGait_SAM3D_NetVLAD(BaseModel):
    def build_network(self, model_cfg):
        # --- Config Parsing ---
        self.image_size = model_cfg["image_size"]
        self.chunk_size = model_cfg.get("chunk_size", 4) # 默认改小一点防止OOM
        self.num_FPN = model_cfg["num_FPN"]
        self.source_dim = model_cfg["source_dim"]
        self.num_parts = model_cfg["num_parts"]   # K=8
        
        # 1. 解析 Hook Mask
        layer_cfg = model_cfg.get("layer_config", {})
        default_mask = [False]*16 + [True]*16 
        self.hook_mask = layer_cfg.get("hook_mask", default_mask)
        
        total_hooked_layers = sum(self.hook_mask)
        assert total_hooked_layers % self.num_FPN == 0, \
            f"Hooked layers ({total_hooked_layers}) must be divisible by num_FPN ({self.num_FPN})"
        
        self.layers_per_fpn = total_hooked_layers // self.num_FPN
        
        # 计算输入总维度: 1280 * Layers_per_FPN * num_FPN
        self.human_conv_input_dim = self.source_dim * self.layers_per_fpn 
        self.total_input_dim = self.human_conv_input_dim * self.num_FPN 

        # GaitNet 每一路需要的维度 (512)
        self.gait_head_dim = model_cfg['SeparateFCs']['in_channels']
        self.total_target_dim = self.gait_head_dim * self.num_FPN # 2048

        # 2. 维度对齐层
        self.dim_reduce = nn.Sequential(
            nn.Conv1d(self.total_input_dim, self.total_target_dim, 1),
            nn.BatchNorm1d(self.total_target_dim),
            nn.LeakyReLU(0.1, inplace=True) # 使用 LeakyReLU 增加鲁棒性
        )

        # 3. NetVLAD Layer
        self.net_vlad = PointNetVLAD(
            feature_size=self.total_target_dim,
            num_clusters=self.num_parts # K=8
        )

        # 4. Gait Recognition Heads
        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)

        # =========================================================
        # 🌟 [新增] 显式冻结 GaitNet 中未使用的 ResNet 层
        # =========================================================
        # 因为我们只调用 test_2 (Head)，跳过了 test_1 (Backbone)
        # 将它们设为 False 后，DDP 就不会检查这些参数的梯度了
        for gait_single in self.Gait_Net.Gait_List:
            # Baseline_Single 包含 pre_rgb 和 post_backbone，这里全部冻结
            for m in [gait_single.pre_rgb, gait_single.post_backbone]:
                for param in m.parameters():
                    param.requires_grad = False

        # 5. SAM Backbone Init
        self.init_SAM_Backbone(model_cfg["pretrained_lvm"])
        self.faces = self.SAM_Engine.head_pose.faces.cpu().numpy()

    def init_SAM_Backbone(self, repo_path):
        sys.path.insert(0, repo_path)
        from notebook.utils import setup_sam_3d_body
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')
        self.SAM_Engine = estimator.model
        self.SAM_Engine.decoder.do_interm_preds = True
        for param in self.SAM_Engine.parameters():
            param.requires_grad = False
        self.SAM_Engine.eval()
        self.Backbone = self.SAM_Engine.backbone
        del estimator

    def init_parameters(self):
        for m in self.dim_reduce.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
        self.msg_mgr.log_info(f"Model Init. Input Dim per Point: {self.total_input_dim} -> VLAD Clusters: {self.num_parts}")

    # --- 辅助函数 ---
    def get_vertex_features(self, image_embeddings, vertices, cam_t, cam_int, img_size):
        B, C, Hf, Wf = image_embeddings.shape
        W_img, H_img = img_size
        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
        z = z.clamp(min=1e-3)
        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy
        u_norm = 2.0 * (u / W_img) - 1.0
        v_norm = 2.0 * (v / H_img) - 1.0
        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(2) 
        sampled_feats = F.grid_sample(
            image_embeddings, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        return sampled_feats.squeeze(-1)
    
    def debug_project_and_viz(self, img_tensor, vertices, cam_t, cam_int, save_path):
        """
        可视化调试函数：将 3D 点投影到 RGB 图上
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 1. 准备数据: Tensor [3, H, W] -> Numpy [H, W, 3]
        img_np = img_tensor.permute(1, 2, 0).detach().cpu().float().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        
        H, W, _ = img_np.shape
        
        # 2. 投影逻辑
        v_cam = vertices + cam_t.unsqueeze(0)
        x, y, z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2].clamp(min=1e-3)
        
        fx, fy = cam_int[0, 0], cam_int[1, 1]
        cx, cy = cam_int[0, 2], cam_int[1, 2]
        
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy
        
        u_np, v_np = u.detach().cpu().numpy(), v.detach().cpu().numpy()
        mask = (u_np >= 0) & (u_np < W) & (v_np >= 0) & (v_np < H)
        
        # 3. 绘图
        fig = plt.figure(figsize=(8, 8 * H / W))
        ax = fig.add_subplot(111)
        ax.imshow(img_np)
        ax.scatter(u_np[mask], v_np[mask], s=12, c='r', alpha=0.6, label='Projected Vertices')
        ax.set_title("3D Projection Alignment Check")
        ax.axis('off')
        
        # 保存文件
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        
        # --- [修复] Matplotlib 3.8+ 兼容写法 ---
        fig.canvas.draw()
        
        # 使用 buffer_rgba() 替代 tostring_rgb()
        # 注意：这里返回的是 RGBA (4通道)，我们需要转为 RGB
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        
        # Reshape: [H, W, 4]
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        # 取前3个通道 (RGB)，丢弃 Alpha
        data_rgb = data[..., :3]
        
        plt.close(fig)
        
        # 返回 Tensor [3, H, W]
        return torch.from_numpy(data_rgb.copy()).float().permute(2, 0, 1) / 255.0
    
    @staticmethod
    def filter_occluded_vertices_strict(vertices, faces, cam_t):
        device = vertices.device
        num_verts = vertices.shape[0]
        v_cam = vertices + cam_t.unsqueeze(0) 
        ray_o = torch.zeros((1, 3), device=device) 
        ray_d = torch.nn.functional.normalize(v_cam, p=2, dim=-1) 
        dist_to_cam = torch.norm(v_cam, p=2, dim=-1)

        v0 = (vertices[faces[:, 0]] + cam_t).unsqueeze(0) 
        v1 = (vertices[faces[:, 1]] + cam_t).unsqueeze(0) 
        v2 = (vertices[faces[:, 2]] + cam_t).unsqueeze(0) 

        is_occluded = torch.zeros(num_verts, dtype=torch.bool, device=device)
        chunk_size = 1024 
        edge1 = v1 - v0
        edge2 = v2 - v0

        for i in range(0, num_verts, chunk_size):
            end_idx = min(i + chunk_size, num_verts)
            curr_ray_d = ray_d[i:end_idx].unsqueeze(1) 
            curr_dist = dist_to_cam[i:end_idx].unsqueeze(1) 

            h = torch.cross(curr_ray_d, edge2, dim=-1)
            a = (edge1 * h).sum(dim=-1)
            mask = a.abs() > 1e-6 

            f = 1.0 / a
            s = ray_o.unsqueeze(1) - v0 
            u = f * (s * h).sum(dim=-1)
            mask &= (u >= 0.0) & (u <= 1.0)
            q = torch.cross(s, edge1, dim=-1)
            v = f * (curr_ray_d * q).sum(dim=-1)
            mask &= (v >= 0.0) & (u + v <= 1.0)

            t = f * (edge2 * q).sum(dim=-1)
            is_hit = mask & (t > 0.01) & (t < curr_dist - 0.01)
            is_occluded[i:end_idx] = is_hit.any(dim=-1)
        return ~is_occluded 

    def filter_and_downsample_to_ply(self, vertices, faces, cam_t, cam_int, img_size, feat_size, step=1, sample_upscale=4):
        device = vertices.device
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces).to(device).long()

        v_cam = vertices + cam_t.unsqueeze(0) 
        x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
        
        fx, fy = cam_int[0, 0], cam_int[1, 1]
        cx, cy = cam_int[0, 2], cam_int[1, 2]
        H_img, W_img  = img_size
        Hf, Wf = (feat_size[0] * sample_upscale, feat_size[1] * sample_upscale) 

        u = (x / (z + 1e-6)) * fx + cx
        v = (y / (z + 1e-6)) * fy + cy
        
        in_frustum = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img) & (z > 0.1)

        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(vertices)
        v_normals.index_add_(0, faces[:, 0], face_normals)
        v_normals.index_add_(0, faces[:, 1], face_normals)
        v_normals.index_add_(0, faces[:, 2], face_normals)
        v_normals = torch.nn.functional.normalize(v_normals, p=2, dim=-1)

        view_dir = torch.nn.functional.normalize(-v_cam, p=2, dim=-1)
        is_front = (v_normals * view_dir).sum(dim=-1) > 0.05
        is_visible_strict = self.filter_occluded_vertices_strict(vertices, faces, cam_t)

        u_feat = (u / W_img * Wf).long().clamp(0, Wf - 1)
        v_feat = (v / H_img * Hf).long().clamp(0, Hf - 1)
        pixel_idx = v_feat * Wf + u_feat 
        
        valid_indices_mask = in_frustum & is_visible_strict & is_front
        valid_indices = torch.where(valid_indices_mask)[0]
        
        if len(valid_indices) == 0:
            return torch.zeros((1, 0, 3), device=device)

        p_idx = pixel_idx[valid_indices]
        p_z = z[valid_indices]
        sorted_inner_idx = torch.argsort(p_z)
        sorted_indices = valid_indices[sorted_inner_idx]
        sorted_pixel_idx = p_idx[sorted_inner_idx]

        _, unique_idx = np.unique(sorted_pixel_idx.cpu().numpy(), return_index=True)
        valid_mask = sorted_indices[unique_idx]
        selected_vertices = vertices[valid_mask]
        downsampled_vertices = selected_vertices[::step]
        return downsampled_vertices.unsqueeze(0)

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

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0] 
        n, s, c, h, w = rgb.size()
        target_h, target_w = self.image_size * 2, self.image_size 
        
        rgb_reshaped = rearrange(rgb, 'n s c h w -> (n s) c h w')
        
        # 为了防显存炸，尽量在循环里做 interpolate，但如果显存够，放外面更快
        # rgb_resized = F.interpolate(rgb_reshaped, (target_h, target_w), mode='bilinear', align_corners=False)
        
        # features_per_person: [n] list of tensors
        features_per_person = [[] for _ in range(n)]
        
        chunk_size = self.chunk_size
        num_chunks = (n * s + chunk_size - 1) // chunk_size

        # 1. 初始化可视化容器
        vis_dict = {}
        has_captured_vis = False
        
        with torch.no_grad():
            for i in range(num_chunks):
                idx_start = i * chunk_size
                idx_end = min((i + 1) * chunk_size, n * s)
                
                # 如果显存不够，在这里 resize
                batch_rgb = F.interpolate(rgb_reshaped[idx_start:idx_end], (target_h, target_w), mode='bilinear', align_corners=False)
                curr_bs = batch_rgb.shape[0]
                
                # A. DINOv3 Features
                all_layers = self.Backbone.encoder.get_intermediate_layers(batch_rgb, n=32, reshape=True, norm=True)
                features_to_use = [f for f, m in zip(all_layers, self.hook_mask) if m]
                sam_emb = all_layers[-1] 
                del all_layers 
                
                # B. SAM 3D Body
                dummy_batch = self._prepare_dummy_batch(sam_emb, target_h, target_w)
                self.SAM_Engine._batch_size = curr_bs
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(curr_bs, device=rgb.device)
                self.SAM_Engine.hand_batch_idx = []
                cond_info = torch.zeros(curr_bs, 3, device=rgb.device); cond_info[:, 2] = 1.1
                dummy_kp = torch.zeros(curr_bs, 1, 3, device=rgb.device); dummy_kp[..., -1] = -2

                with torch.amp.autocast(enabled=False, device_type='cuda'):
                     _, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=sam_emb, keypoints=dummy_kp, condition_info=cond_info, batch=dummy_batch
                    )
                pred_verts_batch = pose_outs[-1]['pred_vertices'] 
                pred_cam_t_batch = pose_outs[-1]['pred_cam_t']    
                cam_int_batch = dummy_batch['cam_int']            

                # C. Per-Sample Processing
                step_per_fpn = len(features_to_use) // self.num_FPN
                
                for b in range(curr_bs):
                    global_idx = idx_start + b
                    person_idx = global_idx // s
                    
                    cur_verts = pred_verts_batch[b] 
                    cur_cam_t = pred_cam_t_batch[b] 
                    cur_cam_int = cam_int_batch[b]  
                    
                    filtered_verts = self.filter_and_downsample_to_ply(
                        cur_verts, self.faces, cur_cam_t, cur_cam_int, (target_h, target_w), 
                        (sam_emb.shape[2], sam_emb.shape[3]), step=2, sample_upscale=4
                    )
                    
                    if filtered_verts.shape[1] == 0: 
                        continue
                    
                    fpn_feats_for_this_sample = []
                    for fpn_idx in range(self.num_FPN):
                        sub_feats_list = features_to_use[fpn_idx*step_per_fpn : (fpn_idx+1)*step_per_fpn]
                        sub_app_b = torch.cat([f[b:b+1] for f in sub_feats_list], dim=1) 
                        feats = self.get_vertex_features(
                            sub_app_b, filtered_verts, cur_cam_t.unsqueeze(0), cur_cam_int.unsqueeze(0), (target_w, target_h)
                        )
                        fpn_feats_for_this_sample.append(feats)
                    
                    combined_feats_b = torch.cat(fpn_feats_for_this_sample, dim=1)
                    features_per_person[person_idx].append(combined_feats_b)

                    # =======================================================
                    # 🌟 [插入] 可视化逻辑 (只在 Rank 0 的第一帧触发一次)
                    # =======================================================
                    if torch.distributed.get_rank() == 0 and not has_captured_vis and filtered_verts.shape[1] > 0:
                        try:
                            import matplotlib.pyplot as plt
                            import numpy as np
                            
                            # 1. 保存原始 RGB
                            vis_rgb = batch_rgb[b].detach().cpu()
                            vis_rgb = (vis_rgb - vis_rgb.min()) / (vis_rgb.max() - vis_rgb.min() + 1e-6)
                            vis_dict['image/00_original'] = vis_rgb

                            # 2. 特征图对齐检查 (PCA)
                            # 使用 FPN 第一组特征 sub_feats_list[0] 的第 b 个样本
                            # shape: [1280, Hf, Wf]
                            feat_raw = features_to_use[0][b].detach().cpu().float()
                            _, Hf, Wf = feat_raw.shape
                            
                            # PCA
                            feat_rgb = compute_pca_rgb(feat_raw.permute(1, 2, 0)) # [H, W, 3]
                            feat_rgb_np = feat_rgb.numpy().astype(np.float32)

                            # 重算坐标用于画点
                            # 注意：我们需要过滤后的点 filtered_verts 在特征图上的位置
                            # 这里需要重新做一次 grid_sample 前的坐标计算步骤
                            _v_cam = filtered_verts[0] + cur_cam_t.unsqueeze(0)
                            _x, _y, _z = _v_cam[:, 0], _v_cam[:, 1], _v_cam[:, 2].clamp(min=1e-3)
                            _u = (_x / _z) * cur_cam_int[0, 0] + cur_cam_int[0, 2]
                            _v = (_y / _z) * cur_cam_int[1, 1] + cur_cam_int[1, 2]
                            
                            # 归一化坐标转特征图坐标
                            # u_norm = 2 * u / W - 1 -> feat = (u_norm + 1)/2 * Wf - 0.5
                            # 简化: feat = u / W * Wf - 0.5 (不考虑 align_corners 的细微差别，可视化够用了)
                            feat_u = (_u / target_w * Wf - 0.5).detach().cpu().numpy()
                            feat_v = (_v / target_h * Hf - 0.5).detach().cpu().numpy()

                            # 绘图 PCA
                            fig = plt.figure(figsize=(8, 8))
                            ax = fig.add_subplot(111)
                            ax.imshow(feat_rgb_np, extent=[-0.5, Wf-0.5, Hf-0.5, -0.5], interpolation='nearest', origin='upper')
                            
                            # 绘制红点
                            mask = (feat_u >= -0.5) & (feat_u < Wf-0.5) & (feat_v >= -0.5) & (feat_v < Hf-0.5)
                            ax.scatter(feat_u[mask], feat_v[mask], s=12, c='red', alpha=0.6, edgecolors='none')
                            ax.set_title("Feature Alignment (FPN Layer 0)")

                            # DEBUG保存文件(时间戳命名）
                            import time
                            time_stamp = int(time.time())
                            plt.savefig(f"debug_feature_pca_align_{time_stamp}.png", bbox_inches='tight', pad_inches=0)
                            
                            # 转 Tensor
                            # --- [修复] Matplotlib 3.8+ 兼容写法 ---
                            fig.canvas.draw()
                            
                            # 1. 获取 RGBA buffer
                            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                            
                            # 2. Reshape [H, W, 4]
                            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                            
                            # 3. Drop Alpha -> [H, W, 3]
                            data_rgb = data[..., :3]
                            
                            plt.close(fig)
                            
                            vis_dict['image/02_feature_pca_align'] = torch.from_numpy(data_rgb.copy()).float().permute(2, 0, 1) / 255.0

                            # 3. 3D 投影对齐检查
                            vis_dict['image/01_projection_align'] = self.debug_project_and_viz(
                                batch_rgb[b], filtered_verts[0], cur_cam_t, cur_cam_int, "debug_proj_latest.png"
                            )
                            
                            has_captured_vis = True
                            print("[DEBUG] Visualization captured successfully.")

                        except Exception as e:
                            print(f"[DEBUG ERROR] Visualization failed: {e}")
                            import traceback
                            traceback.print_exc()

        # =========================================================
        # 2. NetVLAD
        # =========================================================
        parts_features_list = []
        for p_idx, p_list in enumerate(features_per_person):
            if len(p_list) > 0:
                person_merged_feats = torch.cat(p_list, dim=2)
            else:
                person_merged_feats = torch.zeros(1, self.total_input_dim, 1, device=rgb.device)
            
            proj_feats = self.dim_reduce(person_merged_feats)
            vlad_out = self.net_vlad(proj_feats) # [1, 2048, K]
            parts_features_list.append(vlad_out)
            
        parts_features = torch.cat(parts_features_list, dim=0) # [n, 2048, K]
        
        # =========================================================
        # 3. GaitNet Bridge
        # =========================================================
        # Reshape: [n, C, 1, K, 1] -> HPP sees H=K
        gait_input = parts_features.unsqueeze(2).unsqueeze(-1)
        # [核心修复] 传 None 给 seqL
        # 因为我们的 dim=2 已经是 1 了，不需要 TP 根据 seqL 去切片，
        # 传 None 会触发 TP 直接对 dim=2 做 reduction (去掉维度)，这正是我们需要的。
        embed_list, log_list = self.Gait_Net.test_2(gait_input, None)
        
        final_embed = torch.cat(embed_list, dim=1)
        final_logits = torch.cat(log_list, dim=1)

        if self.training:
            return {
                'training_feat': {
                    'triplet': {'embeddings': final_embed, 'labels': labs},
                    'softmax': {'logits': final_logits, 'labels': labs},
                },
                'visual_summary': vis_dict,
                'inference_feat': {'embeddings': final_embed}
            }
        else:
             return {'inference_feat': {'embeddings': final_embed}}