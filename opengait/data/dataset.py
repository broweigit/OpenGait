import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr

import torch
import torch.nn.functional as F

def remove_black_border_batch(batch_imgs, threshold=10, target_h=256, target_w=128):
    """
    批量去除黑边，支持 B x C x H x W 的输入。
    输入: batch_imgs - B x C x H x W 的 PyTorch Tensor，值范围[0,255]或[0,1]
    输出: 去除黑边并调整大小到 target_h x target_w 的 Tensor - B x C x target_h x target_w
    """
    assert batch_imgs.dim() == 4, "Input must be a 4D tensor (B x C x H x W)"
    B, C, H, W = batch_imgs.shape
    batch_imgs = batch_imgs.float()

    # 批量灰度化：对 RGB 取均值，得到 B x H x W
    if C > 1:
        gray = batch_imgs.mean(dim=1)  # B x H x W
    else:
        gray = batch_imgs.squeeze(1)  # B x H x W

    # 批量生成掩码：判断非黑区域
    mask = gray > threshold  # B x H x W

    # 处理全黑图像：直接返回原始图像（或填充为目标尺寸）
    result = torch.zeros(B, C, target_h, target_w, device=batch_imgs.device, dtype=batch_imgs.dtype)
    for i in range(B):
        if not mask[i].any():
            result[i] = resize_with_padding(batch_imgs[i], target_h, target_w)
            continue

        # 批量边界检测
        y_nonzero = torch.any(mask[i], dim=1)  # H
        x_nonzero = torch.any(mask[i], dim=0)  # W

        y_indices = torch.where(y_nonzero)[0]
        x_indices = torch.where(x_nonzero)[0]

        if y_indices.numel() == 0 or x_indices.numel() == 0:
            result[i] = resize_with_padding(batch_imgs[i], target_h, target_w)
            continue

        y_min, y_max = y_indices[[0, -1]]
        x_min, x_max = x_indices[[0, -1]]

        # 裁剪单张图像
        cropped = batch_imgs[i, :, y_min:y_max+1, x_min:x_max+1]
        result[i] = resize_with_padding(cropped, target_h, target_w)

    return result

def resize_with_padding(img, target_h, target_w):
    """
    等比例缩放 + padding，支持单张或批量图像。
    输入: img - C x H x W 或 B x C x H x W
    输出: C x target_h x target_w 或 B x C x target_h x target_w
    """
    is_batch = img.dim() == 4
    if not is_batch:
        img = img.unsqueeze(0)  # 转换为 B x C x H x W

    B, C, H, W = img.shape
    scale = torch.min(torch.tensor([target_h / H, target_w / W], device=img.device))
    new_h, new_w = (H * scale).long(), (W * scale).long()

    # 批量缩放
    img_resized = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # 计算 padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 批量 padding
    img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return img_padded.squeeze(0) if not is_batch else img_padded

# 替换原来的 batch_remove_black_border
def batch_remove_black_border(batch_imgs, threshold=10, target_h=256, target_w=128):
    return remove_black_border_batch(batch_imgs, threshold, target_h, target_w)


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.training = training
        self.video_sample_ratio = data_cfg['video_sample_ratio'] if 'video_sample_ratio' in data_cfg.keys() else None
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        from torchvision.io import read_video
        import random
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
            elif pth.endswith('.avi'):
                video, _, _ = read_video(pth, output_format="TCHW", pts_unit='sec')

                if self.training:
                    random_idx = sorted(random.sample(range(video.size(0)), min(40, video.size(0))), key=int)
                    video = video[random_idx, :, :, :]

                # ratios = torch.full((video.size(0),), video.size(-1) / video.size(-2), device=video.device)
                _ = batch_remove_black_border(video)
                # _ = ratio_resize(_, ratios, 256, 128) # ratio-resize
                _ = _.numpy()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
